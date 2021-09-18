use crate::euler2d_dg;
use crate::mesh;
use crate::node_2d::Cell;
use crate::patch::Patch;
use crate::{ExecutionMode, PatchBasedBuild, PatchBasedSolve, Setup, StructuredMesh};
use cfg_if::cfg_if;
use gpu_core::Device;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton::Automaton;
use gridiron::index_space::{Axis, IndexSpace};
use gridiron::rect_map::Rectangle;
use std::mem::swap;
use std::ops::DerefMut;
use std::os::raw::c_ulong;
use std::sync::{Arc, Mutex};

enum SolverState {
    NotReady,
    RungeKuttaStage(usize),
}

pub struct Solver {
    time: f64,
    time0: f64,
    state: SolverState,
    dt: Option<f64>,
    rk_order: usize,
    weights_rd: Patch,
    weights_wr: Patch,
    // conserved0: Patch,
    // source_buf: Arc<Mutex<Patch>>,
    wavespeeds: Arc<Mutex<Patch>>,
    index_space: IndexSpace,
    // incoming_count: usize,
    // received_count: usize,
    // outgoing_edges: Vec<Rectangle<i64>>,
    cell: Cell,
    mesh: StructuredMesh,
    mode: ExecutionMode,
    device: Option<Device>,
    // setup: Arc<dyn Setup>,
}

impl Solver {
    // pub fn new_timestep(&mut self) {
    //     gpu_core::scope(self.device, || unsafe {
    //         euler2d_dg::euler2d_dg_primitive_to_conserved(
    //             self.mesh,
    //             self.weights_rd.as_ptr(),
    //             self.weights_wr.as_mut_ptr(),
    //             self.mode,
    //         );
    //     });
    //     self.time0 = self.time;
    //     self.state = SolverState::RungeKuttaStage(0);
    // }

    pub fn advance_rk(&mut self, stage: usize) {
        let dt = self.dt.unwrap();

        let a = match self.rk_order {
            1 => match stage {
                0 => 0.0,
                _ => panic!(),
            },
            2 => match stage {
                0 => 0.0,
                1 => 0.5,
                _ => panic!(),
            },
            3 => match stage {
                0 => 0.0,
                1 => 3.0 / 4.0,
                2 => 1.0 / 3.0,
                _ => panic!(),
            },
            _ => panic!(),
        };

        gpu_core::scope(self.device, || unsafe {
            euler2d_dg::euler2d_dg_advance_rk(
                self.cell,
                self.mesh,
                self.weights_rd.as_ptr(),
                self.weights_wr.as_mut_ptr(),
                dt,
                self.mode,
            );
        });
        swap(&mut self.weights_rd, &mut self.weights_wr);

        self.time = self.time0 * a + (self.time + dt) * (1.0 - a);
        self.state = if stage == self.rk_order - 1 {
            SolverState::NotReady
        } else {
            SolverState::RungeKuttaStage(stage + 1)
        }
    }
}

impl PatchBasedSolve for Solver {
    fn primitive(&self) -> Patch {
        todo!("return weights")
        // self.primitive1.extract(&self.index_space)
    }

    fn max_wavespeed(&self) -> f64 {
        let mut lock = self.wavespeeds.lock().unwrap();
        let wavespeeds = lock.deref_mut();

        gpu_core::scope(self.device, || unsafe {
            euler2d_dg::euler2d_dg_wavespeed(
                self.cell,
                self.mesh,
                self.weights_rd.as_ptr(),
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
        todo!()
        // self.outgoing_edges
        //     .iter()
        //     .map(IndexSpace::from)
        //     .map(|neighbor_space| {
        //         let overlap = neighbor_space
        //             .extend_all(2)
        //             .intersect(&self.index_space)
        //             .unwrap();
        //         let guard_patch = self.primitive1.extract(&overlap);
        //         (neighbor_space.to_rect(), guard_patch)
        //     })
        //     .collect()
    }

    fn independent(&self) -> bool {
        // self.incoming_count == 0
        todo!()
    }

    fn receive(&mut self, _neighbor_patch: Self::Message) -> gridiron::automaton::Status {
        todo!()
        // neighbor_patch.copy_into(&mut self.primitive1);
        // self.received_count = (self.received_count + 1) % self.incoming_count;
        // Status::eligible_if(self.received_count == 0)
    }

    fn value(mut self) -> Self::Value {
        if let SolverState::NotReady = self.state {
            todo!()
            // self.new_timestep()
        }
        if let SolverState::RungeKuttaStage(stage) = self.state {
            self.advance_rk(stage)
        }
        self
    }
}

pub struct Builder;

impl PatchBasedBuild for Builder {
    type Solver = Solver;

    fn build(
        &self,
        time: f64,
        primitive: Patch,
        global_structured_mesh: StructuredMesh,
        _edge_list: &AdjacencyList<Rectangle<i64>>,
        rk_order: usize,
        mode: ExecutionMode,
        device: Option<Device>,
        setup: Arc<dyn Setup>,
    ) -> Self::Solver {
        let cell = setup.dg_cell().expect("setup must provide a cell");
        let num_fields = 4 * cell.quadrature_points().count();

        assert! {
            (device.is_none() && std::matches!(mode, ExecutionMode::CPU | ExecutionMode::OMP)) ||
            (device.is_some() && std::matches!(mode, ExecutionMode::GPU)),
            "device must be Some if and only if execution mode is GPU"
        };
        assert_eq! {
            setup.num_primitives(),
            num_fields,
            "this solver requires 4 primitive variable fields, by {} quadrature points",
            cell.quadrature_points().count(),
        };

        let rect = primitive.rect();
        let local_space = primitive.index_space();
        let local_space_ext = local_space.extend_all(2);
        let global_mesh = mesh::Mesh::Structured(global_structured_mesh);
        let global_space_ext = global_mesh.index_space().extend_all(2);

        let guard_spaces = [
            global_space_ext.keep_lower(2, Axis::I),
            global_space_ext.keep_upper(2, Axis::I),
            global_space_ext.keep_lower(2, Axis::J),
            global_space_ext.keep_upper(2, Axis::J),
        ];

        let primitive1 = Patch::zeros(3, &local_space.extend_all(2)).on(device);
        // let conserved0 = Patch::zeros(3, &local_space).on(device);
        // let source_buf = Patch::zeros(3, &locaal_space).on(device);
        let wavespeeds = Patch::zeros(1, &local_space).on(device);

        let mut primitive1 = primitive1;
        primitive.copy_into(&mut primitive1);

        for space in guard_spaces {
            if let Some(overlap) = space.intersect(&local_space_ext) {
                setup
                    .initial_primitive_patch(&overlap, &global_mesh)
                    .copy_into(&mut primitive1);
            }
        }

        Solver {
            time,
            time0: time,
            state: SolverState::NotReady,
            dt: None,
            rk_order,
            weights_rd: primitive1.clone(),
            weights_wr: primitive1,
            // conserved0,
            // source_buf: Arc::new(Mutex::new(source_buf)),
            wavespeeds: Arc::new(Mutex::new(wavespeeds)),
            // outgoing_edges: edge_list.outgoing_edges(&rect).cloned().collect(),
            // incoming_count: edge_list.incoming_edges(&rect).count(),
            // received_count: 0,
            index_space: local_space,
            mode,
            device,
            cell,
            mesh: global_structured_mesh.sub_mesh(rect.0, rect.1),
            // setup,
        }
    }
}
