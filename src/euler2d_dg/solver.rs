use crate::euler2d_dg;
use crate::mesh;
use crate::node_2d::Cell;
use crate::patch::Patch;
use crate::{ExecutionMode, PatchBasedBuild, PatchBasedSolve, Setup, StructuredMesh};
use cfg_if::cfg_if;
use gpu_core::Device;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton::{Automaton, Status};
use gridiron::index_space::{Axis, IndexSpace};
use gridiron::rect_map::Rectangle;
use std::mem::swap;
use std::ops::DerefMut;
use std::os::raw::c_ulong;
use std::sync::{Arc, Mutex};

static NUM_GUARD: usize = 1;
static NUM_CONS: usize = 4;

enum SolverState {
    PreSlopeLimiting,
    PostSlopeLimiting,
}

pub struct Solver {
    time: f64,
    state: SolverState,
    dt: Option<f64>,
    weights1: Patch,
    weights2: Patch,
    wavespeeds: Arc<Mutex<Patch>>,
    index_space: IndexSpace,
    incoming_count: usize,
    received_count: usize,
    outgoing_edges: Vec<Rectangle<i64>>,
    cell: Cell,
    mesh: StructuredMesh,
    mode: ExecutionMode,
    device: Option<Device>,
}

impl Solver {
    pub fn limit_slopes(&mut self) {
        assert!(matches!(self.state, SolverState::PreSlopeLimiting));

        gpu_core::scope(self.device, || unsafe {
            euler2d_dg::euler2d_dg_limit_slopes(
                self.cell,
                self.mesh,
                self.weights1.as_ptr(),
                self.weights2.as_mut_ptr(),
                self.mode,
            );
        });
        swap(&mut self.weights1, &mut self.weights2);
        self.state = SolverState::PostSlopeLimiting;
    }

    pub fn advance_weights(&mut self) {
        assert!(matches!(self.state, SolverState::PostSlopeLimiting));
        let dt = self.dt.unwrap();

        gpu_core::scope(self.device, || unsafe {
            euler2d_dg::euler2d_dg_advance_rk(
                self.cell,
                self.mesh,
                self.weights1.as_ptr(),
                self.weights2.as_mut_ptr(),
                dt,
                self.mode,
            );
        });
        swap(&mut self.weights1, &mut self.weights2);

        self.time = self.time + dt;
        self.state = SolverState::PreSlopeLimiting;
    }
}

impl PatchBasedSolve for Solver {
    fn primitive(&self) -> Patch {
        self.weights1.extract(&self.index_space)
    }

    fn max_wavespeed(&self) -> f64 {
        let mut lock = self.wavespeeds.lock().unwrap();
        let wavespeeds = lock.deref_mut();

        gpu_core::scope(self.device, || unsafe {
            euler2d_dg::euler2d_dg_wavespeed(
                self.cell,
                self.mesh,
                self.weights1.as_ptr(),
                wavespeeds.as_mut_ptr(),
                self.mode,
            )
        });

        match self.mode {
            ExecutionMode::CPU | ExecutionMode::OMP => unsafe {
                euler2d_dg::euler2d_dg_maximum(
                    wavespeeds.as_ptr(),
                    wavespeeds.as_slice().unwrap().len() as c_ulong,
                    self.mode,
                )
            },
            ExecutionMode::GPU => {
                cfg_if! {
                    if #[cfg(feature = "gpu")] {
                        use gpu_core::Reduce;
                        wavespeeds.as_device_buffer().unwrap().maximum().unwrap()
                    } else {
                        unreachable!()
                    }
                }
            }
        }
    }

    fn reductions(&self) -> Vec<f64> {
        vec![]
    }

    fn set_timestep(&mut self, dt: f64) {
        self.dt = Some(dt)
    }

    fn device(&self) -> Option<Device> {
        self.device
    }
}

impl Automaton for Solver {
    type Key = gridiron::rect_map::Rectangle<i64>;
    type Value = Self;
    type Message = Patch;

    fn key(&self) -> Self::Key {
        self.index_space.to_rect()
    }

    fn messages(&self) -> Vec<(Self::Key, Self::Message)> {
        self.outgoing_edges
            .iter()
            .map(IndexSpace::from)
            .map(|neighbor_space| {
                let overlap = neighbor_space
                    .extend_all(NUM_GUARD as i64)
                    .intersect(&self.index_space)
                    .unwrap();
                let guard_patch = self.weights1.extract(&overlap);
                (neighbor_space.to_rect(), guard_patch)
            })
            .collect()
    }

    fn independent(&self) -> bool {
        self.incoming_count == 0
    }

    fn receive(&mut self, neighbor_patch: Self::Message) -> gridiron::automaton::Status {
        neighbor_patch.copy_into(&mut self.weights1);
        self.received_count = (self.received_count + 1) % self.incoming_count;
        Status::eligible_if(self.received_count == 0)
    }

    fn value(mut self) -> Self::Value {
        match self.state {
            SolverState::PreSlopeLimiting => self.limit_slopes(),
            SolverState::PostSlopeLimiting => self.advance_weights(),
        }
        self
    }
}

pub struct Builder;

impl PatchBasedBuild for Builder {
    type Solver = Solver;

    fn stages_per_rk_step(&self) -> usize {
        2
    }

    fn build(
        &self,
        time: f64,
        weights: Patch,
        global_structured_mesh: StructuredMesh,
        edge_list: &AdjacencyList<Rectangle<i64>>,
        rk_order: usize,
        mode: ExecutionMode,
        device: Option<Device>,
        setup: Arc<dyn Setup>,
    ) -> Self::Solver {
        let cell = setup.dg_cell().expect("setup must provide a cell");
        let num_fields = NUM_CONS * cell.num_polynomials();
        let num_guard = NUM_GUARD as i64;

        assert! {
            (device.is_none() && std::matches!(mode, ExecutionMode::CPU | ExecutionMode::OMP)) ||
            (device.is_some() && std::matches!(mode, ExecutionMode::GPU)),
            "device must be Some if and only if execution mode is GPU"
        };
        assert_eq! {
            rk_order, 1, "this solver is hard-coded for RK1 time advance"
        };
        assert_eq! {
            setup.num_primitives() * cell.num_polynomials(),
            num_fields,
            "this solver requires {} conserved variable fields, by {} basis polynomials",
            NUM_CONS,
            cell.num_polynomials(),
        };

        let rect = weights.rect();
        let local_space = weights.index_space();
        let local_space_ext = local_space.extend_all(num_guard);
        let global_mesh = mesh::Mesh::Structured(global_structured_mesh);
        let global_space_ext = global_mesh.index_space().extend_all(num_guard);

        let guard_spaces = [
            global_space_ext.keep_lower(num_guard, Axis::I),
            global_space_ext.keep_upper(num_guard, Axis::I),
            global_space_ext.keep_lower(num_guard, Axis::J),
            global_space_ext.keep_upper(num_guard, Axis::J),
        ];

        let mut weights1 = Patch::zeros(num_fields, &local_space.extend_all(num_guard)).on(device);
        let wavespeeds = Patch::zeros(1, &local_space).on(device);

        weights.copy_into(&mut weights1);

        for space in guard_spaces {
            if let Some(overlap) = space.intersect(&local_space_ext) {
                setup
                    .initial_primitive_patch(&overlap, &global_mesh)
                    .copy_into(&mut weights1);
            }
        }

        Solver {
            time,
            state: SolverState::PreSlopeLimiting,
            dt: None,
            weights2: weights1.clone(),
            weights1,
            wavespeeds: Arc::new(Mutex::new(wavespeeds)),
            outgoing_edges: edge_list.outgoing_edges(&rect).cloned().collect(),
            incoming_count: edge_list.incoming_edges(&rect).count(),
            received_count: 0,
            index_space: local_space,
            mode,
            device,
            cell,
            mesh: global_structured_mesh.sub_mesh(rect.0, rect.1),
        }
    }
}
