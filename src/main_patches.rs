use crate::cmdline;
use crate::error;
use crate::iso2d;
use crate::mesh::Mesh;
use crate::patch::Patch;
use crate::sailfish::ExecutionMode;
use crate::setup::{self, Explosion, Setup};
use crate::split_at_first_colon;
use crate::state::{RecurringTask, State};
use crate::CommandLine;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton;
use gridiron::rect_map::{Rectangle, RectangleMap};
use iso2d::solver::Solver;
use rayon::prelude::*;
use std::sync::Arc;

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

fn make_state(cmdline: &CommandLine) -> Result<State, error::Error> {
    let state = if let Some(ref setup_string) = cmdline.setup {
        let (name, parameters) = split_at_first_colon(setup_string);
        if name.ends_with(".sf") {
            State::from_checkpoint(name, parameters)?
        } else {
            new_state(cmdline.clone(), name, parameters)?
        }
    } else {
        return Err(setup::possible_setups_info());
    };
    if cmdline.upsample.unwrap_or(false) {
        Ok(state.upsample())
    } else {
        Ok(state)
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

    let rk_order = cmdline.rk_order as usize;
    let fold = cmdline.fold;
    let cpi = cmdline.checkpoint_interval;
    let cfl = cmdline.cfl_number;
    let outdir = cmdline.outdir.as_deref().unwrap_or(".");
    let mut state = make_state(&cmdline)?;
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
        Mesh::FacePositions1D(_) => todo!(),
    };

    for (_, patch) in patch_map.into_iter() {
        let solver = Solver::new(
            state.time,
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
