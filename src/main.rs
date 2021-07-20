#[cfg(feature = "omp")]
extern crate openmp_sys;

use crate::cmdline::CommandLine;
use crate::error::Error::*;
use crate::setup::Setup;
use crate::state::State;

use cfg_if::cfg_if;
use std::fmt::Write;
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
        ),
        ("shocktube" | "sedov" | "collision", mesh::Mesh::FacePositions1D(faces)) => euler1d::solver(
            cmdline.execution_mode(),
            cmdline.device,
            faces,
            &state.primitive,
            setup.coordinate_system(),
        ),
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

    while state.time < cmdline.end_time.or_else(|| setup.end_time()).unwrap_or(f64::MAX) {
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

fn main() {
    match run() {
        Ok(_) => {}
        Err(e) => print!("{}", e),
    }
}
