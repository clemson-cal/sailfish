#[cfg(feature = "omp")]
extern crate openmp_sys;
use crate::cmdline::CommandLine;
use crate::error::Error::*;
use crate::mesh::Mesh;
use crate::patch::Patch;
use crate::sailfish::{ExecutionMode, PatchBasedBuild, PatchBasedSolve};
use crate::setup::Setup;
use crate::state::{RecurringTask, State};
use cfg_if::cfg_if;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton;
use gridiron::rect_map::{Rectangle, RectangleMap};
use rayon::prelude::*;
use std::path::Path;

pub mod cmdline;
pub mod error;
pub mod euler1d;
pub mod euler2d;
pub mod iso2d;
pub mod lookup_table;
pub mod mesh;
pub mod patch;
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

pub fn split_at_first(string: &str, character: char) -> (Option<&str>, Option<&str>) {
    let mut a = string.splitn(2, character);
    let n = a.next();
    let p = a.next();
    (n, p)
}

fn parent_dir(path: &str) -> Option<&str> {
    Path::new(path).parent().and_then(Path::to_str)
}

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

fn new_state(
    command_line: CommandLine,
    setup_name: &str,
    parameters: &str,
) -> Result<State, error::Error> {
    let setup = setup::make_setup(setup_name, parameters)?;
    let mesh = setup.mesh(command_line.resolution.unwrap_or(1024));
    let num_patches = match command_line.execution_mode() {
        ExecutionMode::CPU => 512, // TODO: get patch count from command line
        ExecutionMode::OMP => 1,
        ExecutionMode::GPU => gpu_core::all_devices().count(),
    };

    let primitive_patches = match mesh {
        Mesh::Structured(_) => mesh
            .index_space()
            .tile(num_patches)
            .into_iter()
            .map(|s| setup.initial_primitive_patch(&s, &mesh))
            .collect(),
        Mesh::FacePositions1D(_) => vec![],
    };

    let primitive = match mesh {
        Mesh::Structured(_) => vec![],
        Mesh::FacePositions1D(_) => setup.initial_primitive_vec(&mesh),
    };

    let state = State {
        command_line,
        mesh: mesh.clone(),
        restart_file: None,
        iteration: 0,
        time: setup.initial_time(),
        primitive,
        primitive_patches,
        checkpoint: RecurringTask::new(),
        setup_name: setup_name.to_string(),
        parameters: parameters.to_string(),
    };
    Ok(state)
}

fn make_state(cline: &CommandLine) -> Result<State, error::Error> {
    let state = if let Some(ref setup_string) = cline.setup {
        let (name, parameters) = split_at_first(setup_string, ':');
        let (name, parameters) = (name.unwrap_or(""), parameters.unwrap_or(""));
        if name.ends_with(".sf") {
            State::from_checkpoint(name, parameters)?
        } else {
            new_state(cline.clone(), name, parameters)?
        }
    } else {
        return Err(setup::possible_setups_info());
    };
    if cline.upsample.unwrap_or(false) {
        Ok(state.upsample())
    } else {
        Ok(state)
    }
}

fn max_wavespeed<Solver: PatchBasedSolve>(
    solvers: &[Solver],
    pool: &Option<rayon::ThreadPool>,
) -> f64 {
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

fn launch_patch_based<Builder, Solver>(
    mut state: State,
    cline: CommandLine,
    builder: Builder,
) -> Result<(), error::Error>
where
    Builder: PatchBasedBuild<Solver = Solver>,
    Solver: PatchBasedSolve,
{
    let rk_order = cline.rk_order as usize;
    let fold = cline.fold;
    let cpi = cline.checkpoint_interval;
    let cfl = cline.cfl_number;
    let outdir = cline.outdir.as_deref().unwrap_or(".");
    let recompute_dt_each_iteration = cline.recompute_dt_each_iteration()?;
    let setup = setup::make_setup(&state.setup_name, &state.parameters)?;
    let end_time = cline
        .end_time
        .or_else(|| setup.end_time())
        .unwrap_or(f64::MAX);

    let patch_map: RectangleMap<_, _> = state
        .primitive_patches
        .iter()
        .map(|p| (p.rect(), p.clone()))
        .collect();

    let min_spacing = state.mesh.min_spacing();
    let edge_list = adjacency_list(&patch_map, 2);
    let mut solvers = vec![];
    let mut devices = gpu_core::all_devices().cycle();

    let structured_mesh = match state.mesh {
        Mesh::Structured(mesh) => mesh,
        Mesh::FacePositions1D(_) => panic!("the patch-based solver requires a StructuredMesh"),
    };

    for (_, patch) in patch_map.into_iter() {
        let solver = builder.build(
            state.time,
            patch,
            structured_mesh,
            &edge_list,
            rk_order,
            cline.execution_mode(),
            devices.next().filter(|_| cline.use_gpu),
            setup.clone(),
        );
        solvers.push(solver)
    }

    if let Some(mut resolution) = cline.resolution {
        if cline.upsample.unwrap_or(false) {
            resolution *= 2
        }
        if setup.mesh(resolution) != state.mesh {
            return Err(InvalidSetup(
                "cannot override domain parameters on restart".to_string(),
            ));
        }
    }

    if cline.checkpoint_logspace.unwrap_or(false) && setup.initial_time() <= 0.0 {
        return Err(InvalidSetup(
            "checkpoints can only be log-spaced if the initial time is > 0.0".to_string(),
        ));
    }
    setup.print_parameters();

    let pool: Option<rayon::ThreadPool> = match cline.execution_mode() {
        ExecutionMode::CPU => Some(rayon::ThreadPoolBuilder::new().build().unwrap()),
        ExecutionMode::OMP => None,
        ExecutionMode::GPU => None,
    };

    let set_timestep = |solvers: &mut [Solver]| {
        let dt = cfl * min_spacing / max_wavespeed(solvers, &pool);
        for solver in solvers {
            solver.set_timestep(dt)
        }
        dt
    };

    while state.time < end_time {
        if state.checkpoint.is_due(state.time, cpi) {
            state.primitive_patches = solvers.iter().map(|s| s.primitive()).collect();
            state.write_checkpoint(&outdir)?;
        }

        let start = std::time::Instant::now();
        let mut dt = 0.0;

        if !recompute_dt_each_iteration {
            dt = set_timestep(&mut solvers);
        }

        for _ in 0..fold {
            if recompute_dt_each_iteration {
                dt = set_timestep(&mut solvers);
            }
            for _ in 0..rk_order {
                solvers = match pool {
                    Some(ref pool) => {
                        pool.scope(|s| automaton::execute_rayon(s, solvers).collect())
                    }
                    None => automaton::execute(solvers).collect(),
                };
            }
            state.time += dt;
            state.iteration += 1;
        }
        let mzps =
            (state.mesh.num_total_zones() * fold) as f64 / 1e6 / start.elapsed().as_secs_f64();

        println!(
            "[{}] t={:.3} dt={:.3e} Mzps={:.3}",
            state.iteration, state.time, dt, mzps,
        );
    }

    state.primitive_patches = solvers.iter().map(|s| s.primitive()).collect();
    state.write_checkpoint(outdir)?;

    Ok(())
}

fn launch_single_patch(mut state: State, cline: CommandLine) -> Result<(), error::Error> {
    let setup = setup::make_setup(&state.setup_name, &state.parameters)?;
    let recompute_dt_each_iteration = cline.recompute_dt_each_iteration()?;
    let mut solver = match &state.mesh {
        Mesh::FacePositions1D(faces) => euler1d::solver(
            cline.execution_mode(),
            cline.device,
            faces,
            &state.primitive,
            setup.coordinate_system(),
        )?,
        _ => panic!("the single patch solver assumes you have a FacePositions mesh"),
    };

    let (mesh, cfl, fold, chkpt_interval, rk_order, outdir) = (
        state.mesh.clone(),
        cline.cfl_number,
        cline.fold,
        cline.checkpoint_interval,
        cline.rk_order,
        cline
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
    let mut dt = 0.0;
    let dx_min = mesh.min_spacing();
    let end_time = cline
        .end_time
        .or_else(|| setup.end_time())
        .unwrap_or(f64::MAX);

    if cline.checkpoint_logspace.unwrap_or(false) && setup.initial_time() <= 0.0 {
        return Err(InvalidSetup(
            "checkpoints can only be log-spaced if the initial time is > 0.0".to_string(),
        ));
    }
    setup.print_parameters();

    while state.time < end_time {
        if state.checkpoint.is_due(state.time, chkpt_interval) {
            state.set_primitive(solver.primitive());
            state.write_checkpoint(&outdir)?;
        }

        if !recompute_dt_each_iteration {
            dt = cfl * dx_min / solver.max_wavespeed(state.time, setup.as_ref());
        }

        let elapsed = time_exec(cline.device, || {
            for _ in 0..fold {
                if recompute_dt_each_iteration {
                    dt = cfl * dx_min / solver.max_wavespeed(state.time, setup.as_ref());
                }
                solver.advance(setup.as_ref(), rk_order, state.time, dt);
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

fn run() -> Result<(), error::Error> {
    let cline = cmdline::parse_command_line()?;
    let state = make_state(&cline)?;
    let setup = setup::make_setup(&state.setup_name, &state.parameters)?;
    match setup.solver_name().as_str() {
        "iso2d" => launch_patch_based(state, cline, iso2d::solver::Builder),
        "euler1d" => launch_single_patch(state, cline),
        "euler2d" => launch_patch_based(state, cline, euler2d::solver::Builder),
        _ => panic!("unknown solver name"),
    }
}

fn main() {
    if let Err(e) = run() {
        print!("{}", e)
    }
}
