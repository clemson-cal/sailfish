use crate::error::Error;
use std::fmt::Write;

#[derive(Debug)]
pub struct CommandLine {
    pub use_omp: bool,
    pub use_gpu: bool,
    pub resolution: u32,
    pub fold: usize,
    pub checkpoint_interval: f64,
    pub end_time: f64,
    pub rk_order: u32,
    pub cfl_number: f64,
    pub precompute_flux: bool,
}

pub fn parse_command_line() -> Result<CommandLine, Error> {
    let mut c = CommandLine {
        use_gpu: false,
        use_omp: false,
        resolution: 1024,
        fold: 100,
        checkpoint_interval: 1.0,
        end_time: 1.0,
        rk_order: 1,
        cfl_number: 0.2,
        precompute_flux: false,
    };

    enum State {
        Ready,
        GridResolution,
        Fold,
        Checkpoint,
        EndTime,
        RkOrder,
        CFL,
    }
    let mut state = State::Ready;

    for arg in std::env::args()
        .skip(1)
        .flat_map(|arg| arg.split("=").map(str::to_string).collect::<Vec<_>>())
        .flat_map(|arg| {
            if arg.starts_with("-") && !arg.starts_with("--") && arg.len() > 2 {
                let (a, b) = arg.split_at(2);
                vec![a.to_string(), b.to_string()]
            } else {
                vec![arg.to_string()]
            }
        })
    {
        match state {
            State::Ready => match arg.as_str() {
                "--version" => {
                    return Err(Error::PrintUserInformation(format!("sailfish 0.1.0\n")));
                }
                "-h" | "--help" => {
                    let mut message = String::new();
                    writeln!(message, "usage: sailfish [--version] [--help] <[options]>").unwrap();
                    writeln!(message, "       --version             print the code version number").unwrap();
                    writeln!(message, "       -h|--help             display this help message").unwrap();
                    #[cfg(feature = "omp")]
                    writeln!(message, "       -p|--use-omp          run with OpenMP [OMP_NUM_THREADS]").unwrap();
                    writeln!(message, "       -n|--resolution       grid resolution [1024]").unwrap();
                    writeln!(message, "       -f|--fold             number of iterations between messages").unwrap();
                    writeln!(message, "       -c|--checkpoint       amount of time between writing checkpoints").unwrap();
                    writeln!(message, "       -e|--end-time         simulation end time").unwrap();
                    writeln!(message, "       -r|--rk-order         Rnuge-Kutta integration order [1|2|3]").unwrap();
                    writeln!(message, "       --cfl                 CFL number").unwrap();
                    writeln!(message, "       --precompute-flux     compute and store Godunov fluxes before update").unwrap();
                    return Err(Error::PrintUserInformation(message));
                }
                #[cfg(feature = "omp")]
                "-p"|"--use-omp" => c.use_omp = true,
                "-n"|"--res" => state = State::GridResolution,
                "-f"|"--fold" => state = State::Fold,
                "-c"|"--checkpoint" => state = State::Checkpoint,
                "-e"|"--end-time" => state = State::EndTime,
                "-r"|"--rk-order" => state = State::RkOrder,
                "--cfl" => state = State::CFL,
                "--precompute-flux" =>  c.precompute_flux = true,
                _ => return Err(Error::CommandLineParse(format!("unrecognized option {}", arg))),
            },
            State::GridResolution => match arg.parse() {
                Ok(x) => {
                    c.resolution = x;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("resolution {}: {}", arg, e)));
                }
            },
            State::Fold => match arg.parse() {
                Ok(x) => {
                    c.fold = x;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("fold {}: {}", arg, e)));
                }
            },
            State::Checkpoint => match arg.parse() {
                Ok(x) => {
                    c.checkpoint_interval = x;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("checkpoint {}: {}", arg, e)));
                }
            },
            State::RkOrder => match arg.parse() {
                Ok(x) => {
                    if !(1..=3).contains(&x) {
                        return Err(Error::CommandLineParse("rk-order must be 1, 2, or 3".into()))
                    }
                    c.rk_order = x;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("rk-order {}: {}", arg, e)));
                }
            },
            State::EndTime => match arg.parse() {
                Ok(x) => {
                    c.end_time = x;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("checkpoint {}: {}", arg, e)));
                }
            },
            State::CFL => match arg.parse() {
                Ok(x) => {
                    c.cfl_number = x;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("checkpoint {}: {}", arg, e)));
                }
            },
        }
    }

    if !std::matches!(state, State::Ready) {
        return Err(Error::CommandLineParse(format!("missing argument")));
    }
    Ok(c)
}
