use crate::cmdline;
use crate::error;
use crate::iso2d;
use crate::mesh;
use crate::patch::Patch;
use crate::sailfish::{ExecutionMode, StructuredMesh};
use crate::setup::Explosion;
use crate::setup::Setup;
use crate::state::{self, State};
use cfg_if::cfg_if;
use gpu_core::Device;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton::{execute, execute_rayon, Automaton, Status};
use gridiron::index_space::{range2d, Axis, IndexSpace};
use gridiron::rect_map::{Rectangle, RectangleMap};
use rayon::prelude::*;
use std::mem::swap;
use std::ops::DerefMut;
use std::os::raw::c_ulong;
use std::sync::{Arc, Mutex};

fn adjacency_list(
    patches: &RectangleMap<i64, Patch>,
    num_guard: usize,
) -> AdjacencyList<Rectangle<i64>> {
    let mut edges = AdjacencyList::new();
    for (b, q) in patches.iter() {
        for (a, p) in patches.query_rect(q.index_space().extend_all(num_guard as i64)) {
            if a != b {
                edges.insert(p.rect(), q.rect())
            }
        }
    }
    edges
}

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
    wavespeeds: Arc<Mutex<Patch>>,
    index_space: IndexSpace,
    incoming_count: usize,
    received_count: usize,
    outgoing_edges: Vec<Rectangle<i64>>,
    mesh: StructuredMesh,
    mode: ExecutionMode,
    device: Option<Device>,
    setup: Arc<dyn Setup + Send + Sync>,
}

impl Solver {
    pub fn new(
        time: f64,
        primitive: Patch,
        global_structured_mesh: StructuredMesh,
        edge_list: &AdjacencyList<Rectangle<i64>>,
        rk_order: usize,
        mode: ExecutionMode,
        device: Option<Device>,
        setup: Arc<dyn Setup + Send + Sync>,
    ) -> Self {
        assert! {
            (device.is_none() && std::matches!(mode, ExecutionMode::CPU | ExecutionMode::OMP)) ||
            (device.is_some() && std::matches!(mode, ExecutionMode::GPU)),
            "device must be Some if and only if execution mode is GPU"
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
        let primitive2 = Patch::zeros(3, &local_space.extend_all(2)).on(device);
        let conserved0 = Patch::zeros(3, &local_space).on(device);
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

        Self {
            time,
            time0: time,
            state: SolverState::NotReady,
            dt: None,
            rk_order,
            primitive1,
            primitive2,
            conserved0,
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

    pub fn primitive(&self) -> Patch {
        self.primitive1.extract(&self.index_space)
    }

    pub fn max_wavespeed(&self) -> f64 {
        let setup = &self.setup;
        let eos = setup.equation_of_state();
        let masses = gpu_core::Buffer::Host(self.setup.masses(self.time)).on(self.device);
        let mut lock = self.wavespeeds.lock().unwrap();
        let wavespeeds = lock.deref_mut();

        gpu_core::scope(self.device, || unsafe {
            iso2d::iso2d_wavespeed(
                self.mesh,
                self.primitive1.as_ptr(),
                wavespeeds.as_mut_ptr(),
                eos,
                masses.as_ptr(),
                masses.len() as i32,
                self.mode,
            )
        });

        match self.mode {
            ExecutionMode::CPU | ExecutionMode::OMP => unsafe {
                iso2d::iso2d_maximum(
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

    pub fn new_timestep(&mut self) {
        gpu_core::scope(self.device, || unsafe {
            iso2d::iso2d_primitive_to_conserved(
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
        let masses = gpu_core::Buffer::Host(self.setup.masses(self.time)).on(self.device);
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
            iso2d::iso2d_advance_rk(
                self.mesh,
                self.conserved0.as_ptr(),
                self.primitive1.as_ptr(),
                self.primitive2.as_mut_ptr(),
                self.setup.equation_of_state(),
                self.setup.buffer_zone(),
                masses.as_ptr(),
                masses.len() as i32,
                self.setup.viscosity().unwrap_or(0.0),
                a,
                dt,
                f64::MAX,
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

    pub fn set_timestep(&mut self, dt: f64) {
        self.dt = Some(dt)
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

pub fn max_wavespeed(solvers: &[Solver], pool: &Option<rayon::ThreadPool>) -> f64 {
    if let Some(pool) = pool {
        pool.install(|| {
            solvers
                .par_iter()
                .map(|s| s.max_wavespeed())
                .reduce(|| 0.0, f64::max)
        })
    } else {
        solvers
            .iter()
            .map(|s| s.max_wavespeed())
            .fold(0.0, f64::max)
    }
}

pub fn run() -> Result<(), error::Error> {
    let cmdline = cmdline::parse_command_line()?;
    let setup: Arc<dyn Setup + Send + Sync> = Arc::new(Explosion {});

    let n = cmdline.resolution.unwrap_or(2048) as i64;
    let rk_order = cmdline.rk_order as usize;
    let fold = cmdline.fold;
    let cpi = cmdline.checkpoint_interval;
    let cfl = cmdline.cfl_number;
    let outdir = cmdline.outdir.as_deref().unwrap_or(".");
    let num_patches = match cmdline.execution_mode() {
        ExecutionMode::CPU => 512,
        ExecutionMode::OMP => 1,
        ExecutionMode::GPU => gpu_core::all_devices().count(),
    };
    let structured_mesh = StructuredMesh::centered_square(1.0, n as u32);
    let mesh = mesh::Mesh::Structured(structured_mesh.clone());
    let min_spacing = mesh.min_spacing();
    let patch_map: RectangleMap<_, _> = range2d(0..n, 0..n)
        .tile(num_patches)
        .into_iter()
        .map(|s| (s.to_rect(), setup.initial_primitive_patch(&s, &mesh)))
        .collect();

    let edge_list = adjacency_list(&patch_map, 2);
    let mut solvers = vec![];
    let mut devices = gpu_core::all_devices().cycle();

    for (_rect, patch) in patch_map.into_iter() {
        let solver = Solver::new(
            0.0,
            patch,
            structured_mesh,
            &edge_list,
            rk_order,
            cmdline.execution_mode(),
            devices.next().filter(|_| cmdline.use_gpu),
            setup.clone(),
        );
        solvers.push(solver)
    }

    let mut state = State {
        command_line: cmdline.clone(),
        mesh,
        restart_file: None,
        iteration: 0,
        time: setup.initial_time(),
        primitive: vec![],
        primitive_patches: solvers.iter().map(|s| s.primitive()).collect(),
        checkpoint: state::RecurringTask::new(),
        setup_name: String::new(),
        parameters: String::new(),
    };

    let pool: Option<rayon::ThreadPool> = match cmdline.execution_mode() {
        ExecutionMode::CPU => Some(rayon::ThreadPoolBuilder::new().build().unwrap()),
        ExecutionMode::OMP => None,
        ExecutionMode::GPU => None,
    };

    while state.time < cmdline.end_time.unwrap_or(0.1) {
        if state.checkpoint.is_due(state.time, cpi) {
            state.primitive_patches = solvers.iter().map(|s| s.primitive()).collect();
            state.write_checkpoint(&outdir)?;
        }

        let start = std::time::Instant::now();
        let mut dt = 0.0;

        for _ in 0..fold {
            dt = cfl * min_spacing / max_wavespeed(&solvers, &pool);

            for solver in &mut solvers {
                solver.set_timestep(dt)
            }
            for _ in 0..rk_order {
                solvers = match pool {
                    Some(ref pool) => pool.scope(|s| execute_rayon(s, solvers).collect()),
                    None => execute(solvers).collect(),
                };
            }
            state.time += dt;
            state.iteration += 1;
        }
        let mzps = (n * n * fold as i64) as f64 / 1e6 / start.elapsed().as_secs_f64();

        println!(
            "[{}] t={:.3} dt={:.3e} Mzps={:.3}",
            state.iteration, state.time, dt, mzps,
        );
    }

    state.primitive_patches = solvers.iter().map(|s| s.primitive()).collect();
    state.write_checkpoint(outdir)?;

    Ok(())
}
