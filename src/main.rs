#[cfg(feature = "omp")]
extern crate openmp_sys;

use cfg_if::cfg_if;
use rayon::prelude::*;
use std::sync::Arc;

use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton;
use gridiron::rect_map::{Rectangle, RectangleMap};

use sailfish::error::{self, Error::*};
use sailfish::setups;
use sailfish::{euler1d, euler2d, iso2d, sr1d};
use sailfish::{
    CommandLine, ExecutionMode, Mesh, Patch, PatchBasedBuild, PatchBasedSolve, Recurrence,
    RecurringTask, Setup, State,
};

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
    let setup = setups::make_setup(setup_name, parameters)?;
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
        mesh,
        restart_file: None,
        iteration: 0,
        time: setup.initial_time(),
        primitive,
        primitive_patches,
        checkpoint: RecurringTask::new(),
        time_series: RecurringTask::new(),
        time_series_data: vec![],
        setup_name: setup_name.to_string(),
        parameters: parameters.to_string(),
        masses: setup.masses(setup.initial_time()).to_vec(),
        version: sailfish::sailfish_version(),
    };
    Ok(state)
}

fn make_state(cline: &CommandLine) -> Result<State, error::Error> {
    let state = if let Some(ref setup_string) = cline.setup {
        let (name, parameters) = sailfish::parse::split_pair(setup_string, ':');
        let (name, parameters) = (name.unwrap_or(""), parameters.unwrap_or(""));

        if name.ends_with(".sf") {
            State::from_checkpoint(name, &parameters, cline)?
        } else {
            new_state(cline.clone(), name, &parameters)?
        }
    } else {
        return Err(setups::possible_setups_info());
    };
    if cline.upsample.unwrap_or(false) {
        Ok(state.upsample())
    } else {
        Ok(state)
    }
}

fn global_reduction<Solver: PatchBasedSolve>(solvers: &[Solver]) -> Vec<f64> {
    let patch_reductions: Vec<_> = solvers.iter().map(Solver::reductions).collect();
    let start = vec![0.0; patch_reductions[0].len()];

    patch_reductions.iter().fold(start, |a, b| {
        a.into_iter().zip(b).map(|(a, b)| a + b).collect()
    })
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
    setup: Arc<dyn Setup>,
    cline: CommandLine,
    builder: Builder,
) -> Result<(), error::Error>
where
    Builder: PatchBasedBuild<Solver = Solver>,
    Solver: PatchBasedSolve,
{
    let (cfl, fold, rk_order, checkpoint_rule, time_series_rule, dt_each_iter, end_time, outdir) = (
        cline.cfl_number(),
        cline.fold(),
        cline.rk_order(),
        cline.checkpoint_rule(setup.as_ref()),
        cline.time_series_rule(setup.as_ref()),
        cline.recompute_dt_each_iteration(),
        cline.simulation_end_time(setup.as_ref()),
        cline.output_directory(&state.restart_file),
    );
    let patch_map: RectangleMap<_, _> = state
        .primitive_patches
        .iter()
        .map(|p| (p.rect(), p.clone()))
        .collect();

    let min_spacing = state.mesh.min_spacing();
    let edge_list = adjacency_list(&patch_map, 2);
    let mut solvers = vec![];
    let mut devices = if let Some(device) = cline.device {
        vec![gpu_core::Device::with_id(device).unwrap()]
    } else {
        gpu_core::all_devices().collect::<Vec<_>>()
    }
    .into_iter()
    .cycle()
    .filter(|_| cline.use_gpu());

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
            devices.next(),
            setup.clone(),
        );
        solvers.push(solver)
    }

    if std::matches!(checkpoint_rule, Recurrence::Log(_)) && setup.initial_time() <= 0.0 {
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
        if state.time_series.is_due(state.time, time_series_rule) {
            let mut reductions = global_reduction(&solvers);
            reductions.insert(0, state.time);
            state.time_series_data.push(reductions);
            state.time_series.next(state.time, time_series_rule);
            println!("record time series sample {}", state.time_series_data.len(),);
        }
        if state.checkpoint.is_due(state.time, checkpoint_rule) {
            state.primitive_patches = solvers.iter().map(|s| s.primitive()).collect();
            state.write_checkpoint(setup.as_ref(), &outdir)?
        }

        let start = std::time::Instant::now();
        let mut dt = 0.0;

        if !dt_each_iter {
            dt = set_timestep(&mut solvers);
        }

        for _ in 0..fold {
            if dt_each_iter {
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
            state.iteration,
            state.time / setup.unit_time(),
            dt / setup.unit_time(),
            mzps,
        );
    }

    state.primitive_patches = solvers.iter().map(|s| s.primitive()).collect();
    state.write_checkpoint(setup.as_ref(), &outdir)?;

    Ok(())
}

fn launch_single_patch(
    mut state: State,
    setup: Arc<dyn Setup>,
    cline: CommandLine,
) -> Result<(), error::Error> {
    let (cfl, fold, rk_order, checkpoint_rule, dt_each_iter, end_time, outdir) = (
        cline.cfl_number(),
        cline.fold(),
        cline.rk_order(),
        cline.checkpoint_rule(setup.as_ref()),
        cline.recompute_dt_each_iteration(),
        cline.simulation_end_time(setup.as_ref()),
        cline.output_directory(&state.restart_file),
    );
    let mut solver = match &state.mesh {
        Mesh::FacePositions1D(faces) => 
        match setup.solver_name().as_str() {
            "euler1d" => euler1d::solver(
                cline.execution_mode(),
                cline.device,
                faces,
                &state.primitive,
                setup.coordinate_system(),
            )?,
            "sr1d" => sr1d::solver(
                cline.execution_mode(),
                cline.device,
                faces,
                &state.primitive,
                setup.coordinate_system(),
            )?,
            _ => panic!("unknown solver name"),
        }
        _ => panic!("the single patch solver assumes you have a FacePositions mesh"),
    };

    let mut dt = 0.0;
    let dx_min = state.mesh.min_spacing();

    if std::matches!(checkpoint_rule, Recurrence::Log(_)) && setup.initial_time() <= 0.0 {
        return Err(InvalidSetup(
            "checkpoints can only be log-spaced if the initial time is > 0.0".to_string(),
        ));
    }
    setup.print_parameters();

    while state.time < end_time {
        if state.checkpoint.is_due(state.time, checkpoint_rule) {
            state.set_primitive(solver.primitive());
            state.write_checkpoint(setup.as_ref(), &outdir)?;
        }

        if !dt_each_iter {
            dt = cfl * dx_min / solver.max_wavespeed(state.time, setup.as_ref());
        }

        let elapsed = time_exec(cline.device, || {
            for _ in 0..fold {
                if dt_each_iter {
                    dt = cfl * dx_min / solver.max_wavespeed(state.time, setup.as_ref());
                }
                solver.advance(setup.as_ref(), rk_order as u32, state.time, dt);
                state.time += dt;
                state.iteration += 1;
            }
        });

        let mzps = (state.mesh.num_total_zones() * fold) as f64 / 1e6 / elapsed.as_secs_f64();
        println!(
            "[{}] t={:.3} dt={:.3e} Mzps={:.3}",
            state.iteration,
            state.time / setup.unit_time(),
            dt / setup.unit_time(),
            mzps,
        );
    }
    state.set_primitive(solver.primitive());
    state.write_checkpoint(setup.as_ref(), &outdir)?;
    Ok(())
}

fn run() -> Result<(), error::Error> {
    let cline = sailfish::CommandLine::parse()?;
    let state = make_state(&cline)?;
    let setup = setups::make_setup(&state.setup_name, &state.parameters)?;
    let cline = state.command_line.clone();

    match setup.solver_name().as_str() {
        "iso2d" => launch_patch_based(state, setup, cline, iso2d::solver::Builder),
        "euler1d" => launch_single_patch(state, setup, cline),
        "euler2d" => launch_patch_based(state, setup, cline, euler2d::solver::Builder),
        "sr1d" => launch_single_patch(state, setup, cline),
        _ => panic!("unknown solver name"),
    }
}

fn main() {
    if let Err(e) = run() {
        print!("{}", e)
    }
}
