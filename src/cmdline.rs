use git_version::git_version;
use crate::error::Error;
use std::fmt::Write;

#[derive(Debug)]
pub struct CommandLine {
    pub use_omp: bool,
    pub resolution: u64,
    pub fold: u32,
    pub precompute_flux: bool,
}

pub fn parse_command_line() -> Result<CommandLine, Error> {
    let mut c = CommandLine {
        use_omp: false,
        resolution: 1024,
        fold: 100,
        precompute_flux: false,
    };

    enum State {
        Ready,
        GridResolution,
        Fold,
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
                "-h" | "--help" => {
                    let mut message = String::new();
                    writeln!(message, "usage: sailfish [--version] [--help] <[options]>").unwrap();
                    writeln!(message, "       --version             print the code version number").unwrap();
                    writeln!(message, "       -h | --help           display this help message").unwrap();
                    writeln!(message, "       -p | --use-omp        run with OpenMP [OMP_NUM_THREADS]").unwrap();
                    writeln!(message, "       -n | --resolution     grid resolution [1024]").unwrap();
                    writeln!(message, "       -f | --fold           number of iterations between messages").unwrap();
                    writeln!(message, "       --precompute-flux     compute and store Godunov fluxes before update").unwrap();
                    return Err(Error::PrintUserInformation(message));
                }
                "--version" => {
                    return Err(Error::PrintUserInformation(format!("sailfish 0.1.0 {}\n", git_version!())));
                }
                "-p" | "--use-omp" => {
                    c.use_omp = true;
                }
                "--precompute-flux" => {
                    c.precompute_flux = true;
                }
                "-n" | "--res" => {
                    state = State::GridResolution;
                }
                "-f" | "--fold" => {
                    state = State::Fold;
                }
                _ => {
                    return Err(Error::CommandLineParse(format!("unrecognized option {}", arg)));
                }
            },
            State::GridResolution => match arg.parse() {
                Ok(n) => {
                    c.resolution = n;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("-n | --resolution {}: {}", arg, e)));
                }
            },
            State::Fold => match arg.parse() {
                Ok(f) => {
                    c.fold = f;
                    state = State::Ready;
                }
                Err(e) => {
                    return Err(Error::CommandLineParse(format!("-f | --fold {}: {}", arg, e)));
                }
            },
        }
    }

    if !std::matches!(state, State::Ready) {
        return Err(Error::CommandLineParse(format!("missing argument")));
    }
    Ok(c)
}
