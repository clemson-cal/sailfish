use crate::error::Error;
use crate::sailfish::ExecutionMode;
use std::fmt::Write;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandLine {
    pub use_omp: bool,
    pub use_gpu: bool,
    #[serde(default)]
    pub upsample: bool,
    pub setup: Option<String>,
    pub resolution: Option<u32>,
    pub fold: usize,
    pub checkpoint_interval: f64,
    pub outdir: Option<String>,
    pub end_time: Option<f64>,
    pub rk_order: u32,
    pub cfl_number: f64,
    #[serde(default)]
    pub recompute_timestep: String,
    pub velocity_ceiling: f64,
}

impl CommandLine {
    pub fn execution_mode(&self) -> ExecutionMode {
        if self.use_gpu {
            ExecutionMode::GPU
        } else if self.use_omp {
            ExecutionMode::OMP
        } else {
            ExecutionMode::CPU
        }
    }

    pub fn recompute_dt_each_iteration(&self) -> Result<bool, Error> {
        if self.recompute_timestep.is_empty() {
            Ok(true)
        } else if self.recompute_timestep == "iter" {
            Ok(true)
        } else if self.recompute_timestep == "fold" {
            Ok(false)
        } else {
            return Err(Error::Cmdline("invalid mode for --timestep, expected (iter|fold)".to_owned()))
        }
    }
}

#[rustfmt::skip]
pub fn parse_command_line() -> Result<CommandLine, Error> {
    use Error::*;

    let mut c = CommandLine {
        use_omp: false,
        use_gpu: false,
        upsample: false,
        resolution: None,
        fold: 10,
        checkpoint_interval: 1.0,
        setup: None,
        outdir: None,
        end_time: None,
        rk_order: 1,
        cfl_number: 0.2,
        recompute_timestep: String::from(""),
        velocity_ceiling: 1e16,
    };

    enum State {
        Ready,
        GridResolution,
        Fold,
        Checkpoint,
        EndTime,
        RkOrder,
        Cfl,
        Outdir,
        RecomputeTimestep,
        VelocityCeiling
    }
    let mut state = State::Ready;

    for arg in std::env::args()
        .skip(1)
        .flat_map(|arg| {
            if arg.starts_with('-') {
                arg.split('=').map(str::to_string).collect::<Vec<_>>()
            } else {
                vec![arg]
            }
        })
        .flat_map(|arg| {
            if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 2 {
                let (a, b) = arg.split_at(2);
                vec![a.to_string(), b.to_string()]
            } else {
                vec![arg]
            }
        })
    {
        match state {
            State::Ready => match arg.as_str() {
                "--version" => {
                    return Err(PrintUserInformation("sailfish 0.1.0\n".to_string()));
                }
                "-h" | "--help" => {
                    let mut message = String::new();
                    writeln!(message, "usage: sailfish [setup|chkpt] [--version] [--help] <[options]>").unwrap();
                    writeln!(message, "       --version             print the code version number").unwrap();
                    writeln!(message, "       -h|--help             display this help message").unwrap();
                    #[cfg(feature = "omp")]
                    writeln!(message, "       -p|--use-omp          run with OpenMP (reads OMP_NUM_THREADS)").unwrap();
                    #[cfg(feature = "gpu")]
                    writeln!(message, "       -g|--use-gpu          run with GPU acceleration [-p is ignored]").unwrap();
                    writeln!(message, "       -u|--upsample         upsample the grid resolution by a factor of 2").unwrap();
                    writeln!(message, "       -n|--resolution       grid resolution [1024]").unwrap();
                    writeln!(message, "       -f|--fold             number of iterations between messages [10]").unwrap();
                    writeln!(message, "       --timestep            when to recompute time step ([iter]|fold)").unwrap();
                    writeln!(message, "       -c|--checkpoint       amount of time between writing checkpoints [1.0]").unwrap();
                    writeln!(message, "       -o|--outdir           data output directory [current]").unwrap();
                    writeln!(message, "       -e|--end-time         simulation end time [never]").unwrap();
                    writeln!(message, "       -r|--rk-order         Runge-Kutta integration order ([1]|2|3)").unwrap();
                    writeln!(message, "       -v|--velocity-ceiling component-wise velocity ceiling [1e16]").unwrap();
                    writeln!(message, "       --cfl                 CFL number [0.2]").unwrap();
                    return Err(PrintUserInformation(message));
                }
                #[cfg(feature = "omp")]
                "-p" | "--use-omp" => c.use_omp = true,
                #[cfg(feature = "gpu")]
                "-g" | "--use-gpu" => c.use_gpu = true,
                "-u" | "--upsample" => c.upsample = true,
                "-n" | "--resolution" => state = State::GridResolution,
                "-f" | "--fold" => state = State::Fold,
                "--timestep" => state = State::RecomputeTimestep,
                "-c" | "--checkpoint" => state = State::Checkpoint,
                "-o" | "--outdir" => state = State::Outdir,
                "-e" | "--end-time" => state = State::EndTime,
                "-r" | "--rk-order" => state = State::RkOrder,
                "-v" | "--velocity-ceiling" => state = State::VelocityCeiling,
                "--cfl" => state = State::Cfl,
                _ => {
                    if arg.starts_with('-') {
                        return Err(Cmdline(format!("unrecognized option {}", arg)))
                    } else if c.setup.is_some() {
                        return Err(Cmdline(format!("extra positional argument {}", arg)))
                    } else {
                        c.setup = Some(arg)
                    }
                }
            },
            State::GridResolution => {
                c.resolution = Some(arg.parse().map_err(|e| Cmdline(format!("resolution {}: {}", arg, e)))?);
                state = State::Ready;
            }
            State::Fold => {
                c.fold = arg.parse().map_err(|e| Cmdline(format!("fold {}: {}", arg, e)))?;
                state = State::Ready;
            }
            State::RecomputeTimestep => {
                c.recompute_timestep = arg;
                state = State::Ready;
            }
            State::Checkpoint => {
                c.checkpoint_interval = arg.parse().map_err(|e| Cmdline(format!("checkpoint {}: {}", arg, e)))?;
                state = State::Ready;
            }
            State::Outdir => {
                c.outdir = Some(arg);
                state = State::Ready;
            }
            State::RkOrder => {
                c.rk_order = arg.parse().map_err(|e| Cmdline(format!("rk-order {}: {}", arg, e)))?;
                state = State::Ready;
            }
            State::VelocityCeiling => {
                c.velocity_ceiling = arg.parse().map_err(|e| Cmdline(format!("velocity-ceiling {}: {}", arg, e)))?;
                state = State::Ready;
            }
            State::EndTime => {
                c.end_time = Some(arg.parse().map_err(|e| Cmdline(format!("end-time {}: {}", arg, e)))?);
                state = State::Ready;
            }
            State::Cfl => {
                c.cfl_number = arg.parse().map_err(|e| Cmdline(format!("cfl {}: {}", arg, e)))?;
                state = State::Ready;
            }
        }
    }

    if c.use_omp && c.use_gpu {
        Err(Cmdline("--use-omp (-p) and --use-gpu (-g) are mutually exclusive".to_string()))
    } else if !(1..=3).contains(&c.rk_order) {
        Err(Cmdline("rk-order must be 1, 2, or 3".into()))
    } else if !std::matches!(state, State::Ready) {
        Err(Cmdline("missing argument".to_string()))
    } else {
        Ok(c)
    }
}
