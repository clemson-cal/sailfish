use crate::euler2d;
use crate::mesh;
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
    primitive1: Patch,
    primitive2: Patch,
    conserved0: Patch,
    source_buf: Arc<Mutex<Patch>>,
    wavespeeds: Arc<Mutex<Patch>>,
    index_space: IndexSpace,
    incoming_count: usize,
    received_count: usize,
    outgoing_edges: Vec<Rectangle<i64>>,
    mesh: StructuredMesh,
    mode: ExecutionMode,
    device: Option<Device>,
    setup: Arc<dyn Setup>,
}

impl Solver {
    pub fn new_timestep(&mut self) {
        gpu_core::scope(self.device, || unsafe {
            euler2d::euler2d_primitive_to_conserved(
                self.mesh,
                self.primitive1.as_ptr(),
                self.conserved0.as_mut_ptr(),
                self.mode,
            );
        });
        self.time0 = self.time;
        self.state = SolverState::RungeKuttaStage(0);
    }

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
            euler2d::euler2d_advance_rk(
                self.mesh,
                self.conserved0.as_ptr(),
                self.primitive1.as_ptr(),
                self.primitive2.as_mut_ptr(),
                self.setup.equation_of_state(),
                self.setup.boundary_condition(),
                self.setup.masses(self.time),
                self.setup.viscosity().unwrap_or(0.0),
                a,
                dt,
                self.setup.velocity_ceiling().unwrap_or(1e16),
                self.setup.cooling_coefficient().unwrap_or(0.0),
                self.setup.mach_ceiling().unwrap_or(1e5),
                self.setup.density_floor().unwrap_or(0.0),
                self.setup.pressure_floor().unwrap_or(0.0),
                self.mode,
            );
        });
        swap(&mut self.primitive1, &mut self.primitive2);

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
        self.primitive1.extract(&self.index_space)
    }

    fn max_wavespeed(&self) -> f64 {
        let setup = &self.setup;
        let eos = setup.equation_of_state();
        let mut lock = self.wavespeeds.lock().unwrap();
        let wavespeeds = lock.deref_mut();

        gpu_core::scope(self.device, || unsafe {
            euler2d::euler2d_wavespeed(
                self.mesh,
                self.primitive1.as_ptr(),
                wavespeeds.as_mut_ptr(),
                eos,
                self.mode,
            )
        });

        match self.mode {
            ExecutionMode::CPU | ExecutionMode::OMP => unsafe {
                euler2d::euler2d_maximum(
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
        let mut lock = self.source_buf.lock().unwrap();
        let cons_rate = lock.deref_mut();
        let mut result = vec![];
        let mass_list = self.setup.masses(self.time);

        for mass in mass_list.to_vec() {
            gpu_core::scope(self.device, || unsafe {
                euler2d::euler2d_point_mass_source_term(
                    self.mesh,
                    self.primitive1.as_ptr(),
                    cons_rate.as_ptr(),
                    mass_list,
                    mass,
                    self.mode,
                )
            });
            let mut udot = cons_rate
                .to_host()
                .as_slice()
                .unwrap()
                .chunks_exact(4)
                .fold([0.0, 0.0, 0.0, 0.0], |a: [f64; 4], b: &[f64]| {
                    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
                });
            for ud in &mut udot {
                *ud *= self.mesh.dx * self.mesh.dy;
            }
            result.extend(udot)
        }
        result
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
                    .extend_all(2)
                    .intersect(&self.index_space)
                    .unwrap();
                let guard_patch = self.primitive1.extract(&overlap);
                (neighbor_space.to_rect(), guard_patch)
            })
            .collect()
    }

    fn independent(&self) -> bool {
        self.incoming_count == 0
    }

    fn receive(&mut self, neighbor_patch: Self::Message) -> gridiron::automaton::Status {
        neighbor_patch.copy_into(&mut self.primitive1);
        self.received_count = (self.received_count + 1) % self.incoming_count;
        Status::eligible_if(self.received_count == 0)
    }

    fn value(mut self) -> Self::Value {
        if let SolverState::NotReady = self.state {
            self.new_timestep()
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
        edge_list: &AdjacencyList<Rectangle<i64>>,
        rk_order: usize,
        mode: ExecutionMode,
        device: Option<Device>,
        setup: Arc<dyn Setup>,
    ) -> Self::Solver {
        assert! {
            (device.is_none() && std::matches!(mode, ExecutionMode::CPU | ExecutionMode::OMP)) ||
            (device.is_some() && std::matches!(mode, ExecutionMode::GPU)),
            "device must be Some if and only if execution mode is GPU"
        };
        assert_eq! {
            setup.num_primitives(),
            4,
            "this solver is hard-coded for 4 primitive variable fields"
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

        let primitive1 = Patch::zeros(4, &local_space.extend_all(2)).on(device);
        let conserved0 = Patch::zeros(4, &local_space).on(device);
        let source_buf = Patch::zeros(4, &local_space).on(device);
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
            primitive2: primitive1.clone(),
            primitive1,
            conserved0,
            source_buf: Arc::new(Mutex::new(source_buf)),
            wavespeeds: Arc::new(Mutex::new(wavespeeds)),
            outgoing_edges: edge_list.outgoing_edges(&rect).cloned().collect(),
            incoming_count: edge_list.incoming_edges(&rect).count(),
            received_count: 0,
            index_space: local_space,
            mode,
            device,
            mesh: global_structured_mesh.sub_mesh(rect.0, rect.1),
            setup,
        }
    }
}
