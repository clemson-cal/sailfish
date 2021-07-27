use crate::error::Error;
use crate::sailfish::{self, ExecutionMode};
use std::fmt::Write;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandLine {
    pub use_omp: bool,
    pub use_gpu: bool,
    pub device: Option<i32>,
    pub upsample: Option<bool>,
    pub setup: Option<String>,
    pub resolution: Option<u32>,
    pub fold: usize,
    pub checkpoint_interval: f64,
    pub checkpoint_logspace: Option<bool>,
    pub outdir: Option<String>,
    pub end_time: Option<f64>,
    pub rk_order: u32,
    pub cfl_number: f64,
    pub recompute_timestep: Option<String>,
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
        match self.recompute_timestep.as_deref() {
            None => Ok(true),
            Some("iter") => Ok(true),
            Some("fold") => Ok(false),
            _ => Err(Error::Cmdline(
                "invalid mode for --timestep, expected (iter|fold)".to_owned(),
            )),
        }
    }
}

#[rustfmt::skip]
pub fn parse_command_line() -> Result<CommandLine, Error> {
    use Error::*;

    let mut c = CommandLine {
        use_omp: false,
        use_gpu: false,
        device: None,
        upsample: None,
        resolution: None,
        fold: 10,
        checkpoint_interval: 1.0,
        checkpoint_logspace: None,
        setup: None,
        outdir: None,
        end_time: None,
        rk_order: 1,
        cfl_number: 0.2,
        recompute_timestep: None,
        velocity_ceiling: 1e16,
    };

    enum State {
        Ready,
        Device,
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
    std::convert::identity(State::Device); // black-box
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
                    writeln!(message, "       -p|--use-omp          run with OpenMP (reads OMP_NUM_THREADS)").unwrap();
                    writeln!(message, "       -g|--use-gpu          run with GPU acceleration").unwrap();
                    writeln!(message, "       -d|--device           a device ID to run on ([0]-#gpus)").unwrap();
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
                "-p" | "--use-omp" => c.use_omp = true,
                "-g" | "--use-gpu" => c.use_gpu = true,
                "-d" | "--device" => state = State::Device,
                "-u" | "--upsample" => c.upsample = Some(true),
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
            State::Device => {
                c.device = Some(arg.parse().map_err(|e| Cmdline(format!("device {}: {}", arg, e)))?);
                state = State::Ready;                
            }
            State::GridResolution => {
                c.resolution = Some(arg.parse().map_err(|e| Cmdline(format!("resolution {}: {}", arg, e)))?);
                state = State::Ready;
            }
            State::Fold => {
                c.fold = arg.parse().map_err(|e| Cmdline(format!("fold {}: {}", arg, e)))?;
                state = State::Ready;
            }
            State::RecomputeTimestep => {
                c.recompute_timestep = Some(arg);
                state = State::Ready;
            }
            State::Checkpoint => {
                let mut args = arg.splitn(2, ':');
                c.checkpoint_interval = args.next().unwrap_or("").parse().map_err(|e| Cmdline(format!("checkpoint {}: {}", arg, e)))?;
                c.checkpoint_logspace = match args.next() {
                    Some("log") => Some(true),
                    Some("linear")|None => Some(false),
                    _ => return Err(Cmdline("checkpoint mode must be (log|linear) if given".to_string()))
                };
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

    if c.use_omp && !sailfish::compiled_with_omp() {
        Err(CompiledWithoutOpenMP)
    } else if c.use_gpu && !sailfish::compiled_with_gpu() {
        Err(CompiledWithoutGpu)
    } else if c.use_omp && c.use_gpu {
        Err(Cmdline("--use-omp (-p) and --use-gpu (-g) are mutually exclusive".to_string()))
    } else if !(1..=3).contains(&c.rk_order) {
        Err(Cmdline("rk-order must be 1, 2, or 3".into()))
    } else if !std::matches!(state, State::Ready) {
        Err(Cmdline("missing argument".to_string()))
    } else {
        Ok(c)
    }
}
