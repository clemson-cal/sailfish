use error::Error::*;
use setup::Setup;
use solver::cpu;
#[cfg(feature = "cuda")]
use solver::gpu;
use solver::omp;
use solver::Solve;
use state::State;
use std::fmt::Write;
use std::str::FromStr;

pub mod cmdline;
pub mod error;
pub mod setup;
pub mod solver;
pub mod state;

fn time_exec<F>(mut f: F) -> std::time::Duration
where
    F: FnMut(),
{
    let start = std::time::Instant::now();
    f();
    start.elapsed()
}

fn split_at_first_colon(string: &str) -> (&str, &str) {
    let mut a = string.splitn(2, ":");
    let n = a.next().unwrap_or("");
    let p = a.next().unwrap_or("");
    (n, p)
}

fn possible_setups_info() -> error::Error {
    let mut message = String::new();
    writeln!(message, "specify setup:").unwrap();
    writeln!(message, "    binary").unwrap();
    writeln!(message, "    explosion").unwrap();
    PrintUserInformation(message)
}

fn make_setup(setup_name: &str, parameters: &str) -> Result<Box<dyn Setup>, error::Error> {
    match setup_name {
        "binary" => Ok(Box::new(setup::Binary::from_str(parameters)?)),
        "explosion" => Ok(Box::new(setup::Explosion::from_str(parameters)?)),
        _ => Err(possible_setups_info()),
    }
}

fn new_state(resolution: u32, setup_name: &str, parameters: &str) -> Result<State, error::Error> {
    let setup = make_setup(setup_name, parameters)?;
    let mesh = setup.mesh(resolution);
    Ok(State {
        iteration: 0,
        time: 0.0,
        primitive: setup.initial_primitive_vec(&mesh),
        checkpoint: state::RecurringTask::new(),
        setup_name: setup_name.to_string(),
        parameters: parameters.to_string(),
    })
}

fn run() -> Result<(), error::Error> {
    let cmdline = cmdline::parse_command_line()?;

    let (setup, mut state): (Box<dyn Setup>, State) = if let Some(setup_string) = cmdline.setup {
        let (name, parameters) = split_at_first_colon(&setup_string);
        if name.ends_with(".sf") {
            let state = state::State::from_checkpoint(name, parameters)?;
            let setup = make_setup(&state.setup_name, &state.parameters)?;
            (setup, state)
        } else {
            let state = new_state(cmdline.resolution, name, parameters)?;
            let setup = make_setup(name, parameters)?;
            (setup, state)
        }
    } else {
        return Err(possible_setups_info());
    };

    let mesh = setup.mesh(cmdline.resolution);
    let nu = setup.viscosity().unwrap_or(0.0);
    let eos = setup.equation_of_state();
    let buffer = setup.buffer_zone();
    // let v_max = setup.max_signal_speed().unwrap();
    let cfl = cmdline.cfl_number;
    let fold = cmdline.fold;
    let rk_order = cmdline.rk_order;
    // let dt = f64::min(mesh.dx, mesh.dy) / v_max * cfl;

    setup.print_parameters();

    let primitive = setup.initial_primitive_vec(&mesh);
    let mut mzps_log = vec![];
    let mut solver: Box<dyn Solve> = match (cmdline.use_omp, cmdline.use_gpu) {
        (false, false) => Box::new(cpu::Solver::new(mesh.clone(), primitive)),
        (true, false) => Box::new(omp::Solver::new(mesh.clone(), primitive)),
        (false, true) => {
            #[cfg(feature = "cuda")]
            {
                Box::new(gpu::Solver::new(mesh.clone(), primitive))
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda feature not enabled")
            }
        }
        (true, true) => {
            panic!("omp and gpu cannot be enabled at once")
        }
    };

    while state.time < cmdline.end_time {
        if state.checkpoint.last_time.is_none() || 
           state.time >= state.checkpoint.last_time.unwrap() + cmdline.checkpoint_interval {
            state.write_checkpoint(cmdline.checkpoint_interval, &cmdline.outdir)?;
        }
    
        let elapsed = time_exec(|| {
            for _ in 0..fold {
                let a_max = solver.max_wavespeed(&eos, &setup.masses(state.time));
                let dt = f64::min(mesh.dx, mesh.dy) / a_max * cfl;
                println!("{}", dt);

                solver::advance(
                    &mut solver,
                    &eos,
                    &buffer,
                    |t| setup.masses(t),
                    nu,
                    rk_order,
                    state.time,
                    dt,
                );
                state.time += dt;
                state.iteration += 1;
            }
        });

        mzps_log.push((mesh.num_total_zones() * fold) as f64 / 1e6 / elapsed.as_secs_f64());
        println!(
            "[{}] t={:.3} Mzps={:.3}",
            state.iteration,
            state.time,
            mzps_log.last().unwrap()
        );
    }
    state.write_checkpoint(cmdline.checkpoint_interval, &cmdline.outdir)
}

fn main() {
    match run() {
        Ok(_) => {}
        Err(e) => print!("{}", e),
    }
}
