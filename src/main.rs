use crate::cmdline::CommandLine;
use crate::error::Error::*;
use crate::sailfish::{Mesh, Solve};
use crate::setup::Setup;
use crate::state::State;
use cfg_if::cfg_if;
use std::fmt::Write;
use std::path::Path;
use std::str::FromStr;

pub mod cmdline;
pub mod error;
pub mod iso2d;
pub mod sailfish;
pub mod setup;
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
    PrintUserInformation(message)
}

fn make_setup(setup_name: &str, parameters: &str) -> Result<Box<dyn Setup>, error::Error> {
    match setup_name {
        "binary" => Ok(Box::new(setup::Binary::from_str(parameters)?)),
        "explosion" => Ok(Box::new(setup::Explosion::from_str(parameters)?)),
        _ => Err(possible_setups_info()),
    }
}

fn make_solver(cmdline: &CommandLine, mesh: Mesh, primitive: Vec<f64>) -> Box<dyn Solve> {
    if cmdline.use_gpu {
        cfg_if! {
            if #[cfg(feature = "cuda")] {
                Box::new(iso2d::gpu::Solver::new(mesh, primitive))
            } else {
                panic!()
            }
        }
    } else if cmdline.use_omp {
        cfg_if! {
            if #[cfg(feature = "omp")] {
                Box::new(iso2d::omp::Solver::new(mesh, primitive))
            } else {
                panic!()
            }
        }
    } else {
        Box::new(iso2d::cpu::Solver::new(mesh, primitive))
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
        mesh,
        restart_file: None,
        iteration: 0,
        time: 0.0,
        primitive: setup.initial_primitive_vec(&mesh),
        checkpoint: state::RecurringTask::new(),
        setup_name: setup_name.to_string(),
        parameters: parameters.to_string(),
    };
    Ok(state)
}

fn make_state(cmdline: &CommandLine) -> Result<State, error::Error> {
    if let Some(ref setup_string) = cmdline.setup {
        let (name, parameters) = split_at_first_colon(&setup_string);
        if name.ends_with(".sf") {
            state::State::from_checkpoint(name, parameters)
        } else {
            new_state(cmdline.clone(), name, parameters)
        }
    } else {
        Err(possible_setups_info())
    }
}

fn parent_dir(path: &str) -> Option<&str> {
    Path::new(path).parent().and_then(Path::to_str)
}

fn run() -> Result<(), error::Error> {
    let cmdline = cmdline::parse_command_line()?;

    let mut state = make_state(&cmdline)?;
    let mut solver = make_solver(&cmdline, state.mesh, state.primitive.clone());
    let mut mzps_log = vec![];

    let setup = make_setup(&state.setup_name, &state.parameters)?;
    let (mesh, nu, eos, buffer, cfl, fold, chkpt_interval, rk_order, outdir) = (
        state.mesh,
        setup.viscosity().unwrap_or(0.0),
        setup.equation_of_state(),
        setup.buffer_zone(),
        cmdline.cfl_number,
        cmdline.fold,
        cmdline.checkpoint_interval,
        cmdline.rk_order,
        cmdline
            .outdir
            .or_else(|| {
                state
                    .restart_file
                    .as_deref()
                    .and_then(parent_dir)
                    .map(String::from)
            })
            .unwrap_or(String::from(".")),
    );

    if let Some(resolution) = cmdline.resolution {
        if setup.mesh(resolution) != mesh {
            return Err(InvalidSetup(
                "cannot override domain parameters on restart".to_string(),
            ));
        }
    }
    println!("outdir: {}", outdir);
    setup.print_parameters();

    while state.time < cmdline.end_time.unwrap_or(f64::MAX) {
        if state.checkpoint.is_due(state.time, chkpt_interval) {
            state.set_primitive(solver.primitive());
            state.write_checkpoint(chkpt_interval, &outdir)?;
        }

        let elapsed = time_exec(|| {
            for _ in 0..fold {
                let a_max = solver.max_wavespeed(eos, &setup.masses(state.time));
                let dt = f64::min(mesh.dx, mesh.dy) / a_max * cfl;

                iso2d::advance(
                    &mut solver,
                    eos,
                    buffer,
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
    state.set_primitive(solver.primitive());
    state.write_checkpoint(chkpt_interval, &outdir)
}

fn main() {
    match run() {
        Ok(_) => {}
        Err(e) => print!("{}", e),
    }
}
