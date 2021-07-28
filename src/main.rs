#[cfg(feature = "omp")]
extern crate openmp_sys;
use std::sync::Arc;
use crate::cmdline::CommandLine;
use crate::error::Error::*;
use crate::setup::Setup;
use crate::state::State;
use cfg_if::cfg_if;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton::{self, Automaton, Status};
use gridiron::index_space::range2d;
use gridiron::index_space::IndexSpace;
use gridiron::patch::Patch;
use gridiron::rect_map::{Rectangle, RectangleMap};
use setup::Explosion;
use std::fmt::Write;
use std::mem::swap;
use std::path::Path;
use std::str::FromStr;

pub mod cmdline;
pub mod error;
pub mod euler1d;
pub mod iso2d;
pub mod lookup_table;
pub mod mesh;
pub mod sailfish;
pub mod setup;
pub mod state;

fn time_exec<F>(device: Option<i32>, mut f: F) -> std::time::Duration
where
    F: FnMut(),
{
    let start = std::time::Instant::now();
    f();

    cfg_if! {
        if #[cfg(feature = "gpu")] {
            gpu_core::Device::with_id(device.unwrap_or(0)).unwrap().synchronize();
        } else {
            std::convert::identity(device); // black-box
        }
    }
    start.elapsed()
}

fn split_at_first_colon(string: &str) -> (&str, &str) {
    let mut a = string.splitn(2, ':');
    let n = a.next().unwrap_or("");
    let p = a.next().unwrap_or("");
    (n, p)
}

fn possible_setups_info() -> error::Error {
    let mut message = String::new();
    writeln!(message, "specify setup:").unwrap();
    writeln!(message, "    binary").unwrap();
    writeln!(message, "    explosion").unwrap();
    writeln!(message, "    shocktube").unwrap();
    writeln!(message, "    collision").unwrap();
    writeln!(message, "    sedov").unwrap();
    PrintUserInformation(message)
}

fn make_setup(setup_name: &str, parameters: &str) -> Result<Box<dyn Setup>, error::Error> {
    use setup::*;
    match setup_name {
        "binary" => Ok(Box::new(Binary::from_str(parameters)?)),
        "explosion" => Ok(Box::new(Explosion::from_str(parameters)?)),
        "shocktube" => Ok(Box::new(Shocktube::from_str(parameters)?)),
        "sedov" => Ok(Box::new(Sedov::from_str(parameters)?)),
        "collision" => Ok(Box::new(Collision::from_str(parameters)?)),
        _ => Err(possible_setups_info()),
    }
}

fn new_state(
    command_line: CommandLine,
    setup_name: &str,
    parameters: &str,
) -> Result<State, error::Error> {
    let setup = make_setup(setup_name, parameters)?;
    let mesh = setup.mesh(command_line.resolution.unwrap_or(1024));

    let state = State {
        command_line,
        mesh: mesh.clone(),
        restart_file: None,
        iteration: 0,
        time: setup.initial_time(),
        primitive: setup.initial_primitive_vec(&mesh),
        primitive_patches: vec![],
        checkpoint: state::RecurringTask::new(),
        setup_name: setup_name.to_string(),
        parameters: parameters.to_string(),
    };
    Ok(state)
}

fn make_state(cmdline: &CommandLine) -> Result<State, error::Error> {
    let state = if let Some(ref setup_string) = cmdline.setup {
        let (name, parameters) = split_at_first_colon(setup_string);
        if name.ends_with(".sf") {
            state::State::from_checkpoint(name, parameters)?
        } else {
            new_state(cmdline.clone(), name, parameters)?
        }
    } else {
        return Err(possible_setups_info());
    };
    if cmdline.upsample.unwrap_or(false) {
        Ok(state.upsample())
    } else {
        Ok(state)
    }
}

fn parent_dir(path: &str) -> Option<&str> {
    Path::new(path).parent().and_then(Path::to_str)
}

fn run() -> Result<(), error::Error> {
    let cmdline = cmdline::parse_command_line()?;
    let mut state = make_state(&cmdline)?;
    let mut dt = 0.0;
    let setup = make_setup(&state.setup_name, &state.parameters)?;
    let recompute_dt_each_iteration = cmdline.recompute_dt_each_iteration()?;
    let mut solver = match (state.setup_name.as_str(), &state.mesh) {
        ("binary" | "explosion", mesh::Mesh::Structured(mesh)) => iso2d::solver(
            cmdline.execution_mode(),
            cmdline.device,
            *mesh,
            &state.primitive,
        )?,
        ("shocktube" | "sedov" | "collision", mesh::Mesh::FacePositions1D(faces)) => {
            euler1d::solver(
                cmdline.execution_mode(),
                cmdline.device,
                faces,
                &state.primitive,
                setup.coordinate_system(),
            )?
        }
        _ => panic!(),
    };

    let (mesh, cfl, fold, chkpt_interval, rk_order, velocity_ceiling, outdir) = (
        state.mesh.clone(),
        cmdline.cfl_number,
        cmdline.fold,
        cmdline.checkpoint_interval,
        cmdline.rk_order,
        cmdline.velocity_ceiling,
        cmdline
            .outdir
            .or_else(|| {
                state
                    .restart_file
                    .as_deref()
                    .and_then(parent_dir)
                    .map(String::from)
            })
            .unwrap_or_else(|| String::from(".")),
    );
    let dx_min = mesh.min_spacing();

    if let Some(mut resolution) = cmdline.resolution {
        if cmdline.upsample.unwrap_or(false) {
            resolution *= 2
        }
        if setup.mesh(resolution) != mesh {
            return Err(InvalidSetup(
                "cannot override domain parameters on restart".to_string(),
            ));
        }
    }

    if cmdline.checkpoint_logspace.unwrap_or(false) && setup.initial_time() <= 0.0 {
        return Err(InvalidSetup(
            "checkpoints can only be log-spaced if the initial time is > 0.0".to_string(),
        ));
    }

    println!("outdir: {}", outdir);
    setup.print_parameters();

    while state.time
        < cmdline
            .end_time
            .or_else(|| setup.end_time())
            .unwrap_or(f64::MAX)
    {
        if state.checkpoint.is_due(state.time, chkpt_interval) {
            state.set_primitive(solver.primitive());
            state.write_checkpoint(&outdir)?;
        }

        if !recompute_dt_each_iteration {
            dt = dx_min / solver.max_wavespeed(state.time, setup.as_ref()) * cfl;
        }

        let elapsed = time_exec(cmdline.device, || {
            for _ in 0..fold {
                if recompute_dt_each_iteration {
                    dt = dx_min / solver.max_wavespeed(state.time, setup.as_ref()) * cfl;
                }
                solver.advance(setup.as_ref(), rk_order, state.time, dt, velocity_ceiling);
                state.time += dt;
                state.iteration += 1;
            }
        });

        let mzps = (mesh.num_total_zones() * fold) as f64 / 1e6 / elapsed.as_secs_f64();
        println!(
            "[{}] t={:.3} dt={:.3e} Mzps={:.3}",
            state.iteration, state.time, dt, mzps,
        );
    }
    state.set_primitive(solver.primitive());
    state.write_checkpoint(&outdir)?;
    Ok(())
}

fn adjacency_list(
    patches: &RectangleMap<i64, Patch>,
    num_guard: usize,
) -> AdjacencyList<Rectangle<i64>> {
    let mut edges = AdjacencyList::new();
    for (b, q) in patches.iter() {
        for (a, p) in patches.query_rect(q.index_space().extend_all(num_guard as i64)) {
            if a != b {
                edges.insert(p.high_resolution_rect(), q.high_resolution_rect())
            }
        }
    }
    edges
}

pub struct Solver {
    time: f64,
    dt: f64,
    primitive1: Patch,
    primitive2: Patch,
    conserved0: Patch,
    index_space: IndexSpace,
    incoming_count: usize,
    received_count: usize,
    outgoing_edges: Vec<Rectangle<i64>>,
    mesh: sailfish::StructuredMesh,
    setup: Arc<dyn Setup + Send + Sync>,
}

impl Solver {
    pub fn new(
        time: f64,
        primitive: Patch,
        global_mesh: sailfish::StructuredMesh,
        edge_list: &AdjacencyList<Rectangle<i64>>,
        setup: Arc<dyn Setup + Send + Sync>,
    ) -> Self {
        let index_space = primitive.high_resolution_space();
        let rect = primitive.high_resolution_rect();
        let key = rect.clone();
        let mesh = global_mesh.sub_mesh(rect.0, rect.1);
        let primitive = Patch::extract_from(&primitive, index_space.extend_all(2));
        Self {
            time,
            dt: 0.0,
            conserved0: Patch::zeros(0, 3, index_space.clone()),
            primitive1: primitive.clone(),
            primitive2: primitive,
            outgoing_edges: edge_list.outgoing_edges(&key).cloned().collect(),
            incoming_count: edge_list.incoming_edges(&key).count(),
            received_count: 0,
            index_space,
            mesh,
            setup,
        }
    }

    pub fn primitive(&self) -> Patch {
        self.primitive1.extract(self.index_space.clone())
    }

    pub fn max_wavespeed(&self) -> f64 {
        let setup = &self.setup;
        let eos = setup.equation_of_state();
        let masses = setup.masses(self.time);
        let mut wavespeeds = vec![0.0; self.mesh.num_total_zones()];
        unsafe {
            iso2d::iso2d_wavespeed(
                self.mesh,
                self.primitive1.data().as_ptr(),
                wavespeeds.as_mut_ptr(),
                eos,
                masses.as_ptr(),
                masses.len() as i32,
                sailfish::ExecutionMode::CPU,
            )
        };
        wavespeeds.iter().cloned().fold(0.0, f64::max)
    }

    pub fn set_timestep(&mut self, dt: f64) {
        self.dt = dt
    }
}

impl Automaton for Solver {
    type Key = gridiron::rect_map::Rectangle<i64>;
    type Value = Self;
    type Message = gridiron::patch::Patch;
    fn key(&self) -> Self::Key {
        self.index_space.clone().into_rect()
    }
    fn messages(&self) -> Vec<(Self::Key, Self::Message)> {
        self.outgoing_edges
            .iter()
            .map(IndexSpace::from)
            .map(|neighbor_space| {
                let overlap = neighbor_space.extend_all(2).intersect(self.index_space.clone());
                let guard_patch = self.primitive1.extract(overlap);
                (neighbor_space.clone().into_rect(), guard_patch)
            })
            .collect()
    }
    fn independent(&self) -> bool {
        self.incoming_count == 0
    }
    fn receive(&mut self, neighbor_patch: Self::Message) -> gridiron::automaton::Status {
        neighbor_patch.copy_into(&mut self.primitive1);
        self.received_count += 1;
        Status::eligible_if(self.received_count == self.incoming_count)
    }
    fn value(mut self) -> Self::Value {
        self.received_count = 0;
        let masses = self.setup.masses(self.time);

        unsafe {
            iso2d::iso2d_advance_rk(
                self.mesh,
                self.conserved0.data().as_ptr(),
                self.primitive1.data().as_ptr(),
                self.primitive2.data_mut().as_mut_ptr(),
                self.setup.equation_of_state(),
                self.setup.buffer_zone(),
                masses.as_ptr(),
                masses.len() as i32,
                self.setup.viscosity().unwrap_or(0.0),
                0.0,
                self.dt,
                f64::MAX,
                sailfish::ExecutionMode::CPU,
            );
        }
        swap(&mut self.primitive1, &mut self.primitive2);
        self
    }
}

fn run_decomposed_domain() -> Result<(), error::Error> {
    let setup = setup::Explosion {};
    let n = 1024;
    let global_mesh = sailfish::StructuredMesh::centered_square(1.0, n as u32);
    let patch_map: RectangleMap<_, _> = range2d(0..n, 0..n)
        .tile(128)
        .into_iter()
        .map(|space| {
            let (i0, j0) = space.start();
            let (di, dj) = space.clone().into_rect();
            let mesh = global_mesh.sub_mesh(di.clone(), dj.clone());

            let patch = Patch::from_slice_function(0, space, 3, |(i, j), prim| {
                let [x, y] = mesh.cell_coordinates(i - i0, j - j0);
                setup.initial_primitive(x, y, prim)
            });
            (patch.high_resolution_rect(), patch)
        })
        .collect();

    let edge_list = adjacency_list(&patch_map, 2);

    let setup: Arc<dyn Setup + Send + Sync> = Arc::new(Explosion{});
    let mut solvers: Vec<_> = patch_map
        .into_iter()
        .map(|(_rect, patch)| Solver::new(0.0, patch, global_mesh, &edge_list, setup.clone()))
        .collect();

    let fold = 10;
    let cfl = 0.2;
    let min_spacing = f64::min(global_mesh.dx, global_mesh.dy);
    let max_a = solvers.iter().map(|solver| solver.max_wavespeed()).fold(0.0, f64::max);
    let dt = cfl * min_spacing / max_a;

    let mut time = 0.0;
    let mut iteration = 0;

    for solver in &mut solvers {
        solver.set_timestep(dt)
    }
    let pool = gridiron::thread_pool::ThreadPool::new(28);

    while time < 0.1 {
        let start = std::time::Instant::now();

        for _ in 0..fold {
            solvers = automaton::execute_thread_pool(&pool, solvers).collect();
            time += dt;
            iteration += 1;
        }
        let mzps = (n * n * fold) as f64 / 1e6 / start.elapsed().as_secs_f64();

        println!(
            "[{}] t={:.3} dt={:.3e} Mzps={:.3}",
            iteration, time, dt, mzps,
        );
    }

    let patches = solvers.into_iter().map(|solver| solver.primitive()).collect();

    let mut state = State {
        command_line: CommandLine::default(),
        mesh: mesh::Mesh::Structured(global_mesh.clone()),
        restart_file: None,
        iteration,
        time,
        primitive: vec![],
        primitive_patches: patches,
        checkpoint: state::RecurringTask::new(),
        setup_name: String::new(),
        parameters: String::new(),
    };
    state.write_checkpoint(".")?;

    Ok(())
}

fn main() {
    if false {
        match run() {
            Ok(_) => {}
            Err(e) => print!("{}", e),
        }
    } else {
        match run_decomposed_domain() {
            Ok(_) => {}
            Err(e) => print!("{}", e),
        }
    }
}
