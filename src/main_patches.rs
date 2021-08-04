use crate::cmdline;
use crate::error;
use crate::iso2d;
use crate::mesh;
use crate::patch::Patch;
use crate::sailfish::{ExecutionMode, StructuredMesh};
use crate::setup::{Setup, Explosion};
use crate::state;
use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton;
use gridiron::index_space::range2d;
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

    let mut state = state::State {
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
                    Some(ref pool) => {
                        pool.scope(|s| automaton::execute_rayon(s, solvers).collect())
                    }
                    None => automaton::execute(solvers).collect(),
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
