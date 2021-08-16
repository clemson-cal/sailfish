use crate::error::Error;
use crate::{ExecutionMode, Setup, Recurrence};
use std::fmt::Write;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandLine {
    pub use_omp: Option<bool>,
    pub use_gpu: Option<bool>,
    pub device: Option<i32>,
    pub upsample: Option<bool>,
    pub setup: Option<String>,
    pub resolution: Option<u32>,
    pub fold: Option<usize>,
    pub checkpoint_interval: Option<f64>,
    pub checkpoint_logspace: Option<bool>,
    pub outdir: Option<String>,
    pub end_time: Option<f64>,
    pub rk_order: Option<usize>,
    pub cfl_number: Option<f64>,
    pub recompute_timestep: Option<String>,
}

impl CommandLine {
    pub fn parse() -> Result<Self, Error> {
        use Error::*;
        let mut c = Self::default();

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
                        return Err(PrintUserInformation("sailfish 0.2.0\n".to_string()));
                    }

                    #[rustfmt::skip]
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
                        writeln!(message, "       --cfl                 CFL number [0.2]").unwrap();
                        return Err(PrintUserInformation(message));
                    }
                    "-p" | "--use-omp" => c.use_omp = Some(true),
                    "-g" | "--use-gpu" => c.use_gpu = Some(true),
                    "-d" | "--device" => state = State::Device,
                    "-u" | "--upsample" => c.upsample = Some(true),
                    "-n" | "--resolution" => state = State::GridResolution,
                    "-f" | "--fold" => state = State::Fold,
                    "--timestep" => state = State::RecomputeTimestep,
                    "-c" | "--checkpoint" => state = State::Checkpoint,
                    "-o" | "--outdir" => state = State::Outdir,
                    "-e" | "--end-time" => state = State::EndTime,
                    "-r" | "--rk-order" => state = State::RkOrder,
                    "--cfl" => state = State::Cfl,
                    _ => {
                        if arg.starts_with('-') {
                            return Err(Cmdline(format!("unrecognized option {}", arg)));
                        } else if c.setup.is_some() {
                            return Err(Cmdline(format!("extra positional argument {}", arg)));
                        } else {
                            c.setup = Some(arg)
                        }
                    }
                },
                State::Device => {
                    c.device = Some(
                        arg.parse()
                            .map_err(|e| Cmdline(format!("device {}: {}", arg, e)))?,
                    );
                    state = State::Ready;
                }
                State::GridResolution => {
                    c.resolution = Some(
                        arg.parse()
                            .map_err(|e| Cmdline(format!("resolution {}: {}", arg, e)))?,
                    );
                    state = State::Ready;
                }
                State::Fold => {
                    c.fold = Some(
                        arg.parse()
                            .map_err(|e| Cmdline(format!("fold {}: {}", arg, e)))?,
                    );
                    state = State::Ready;
                }
                State::RecomputeTimestep => {
                    c.recompute_timestep = Some(arg);
                    state = State::Ready;
                }
                State::Checkpoint => {
                    let mut args = arg.splitn(2, ':');
                    c.checkpoint_interval = Some(
                        args.next()
                            .unwrap_or("")
                            .parse()
                            .map_err(|e| Cmdline(format!("checkpoint {}: {}", arg, e)))?,
                    );
                    c.checkpoint_logspace = match args.next() {
                        Some("log") => Some(true),
                        Some("linear") | None => Some(false),
                        _ => {
                            return Err(Cmdline(
                                "checkpoint mode must be (log|linear) if given".to_string(),
                            ))
                        }
                    };
                    state = State::Ready;
                }
                State::Outdir => {
                    c.outdir = Some(arg);
                    state = State::Ready;
                }
                State::RkOrder => {
                    c.rk_order = Some(
                        arg.parse()
                            .map_err(|e| Cmdline(format!("rk-order {}: {}", arg, e)))?,
                    );
                    state = State::Ready;
                }
                State::EndTime => {
                    c.end_time = Some(
                        arg.parse()
                            .map_err(|e| Cmdline(format!("end-time {}: {}", arg, e)))?,
                    );
                    state = State::Ready;
                }
                State::Cfl => {
                    c.cfl_number = Some(
                        arg.parse()
                            .map_err(|e| Cmdline(format!("cfl {}: {}", arg, e)))?,
                    );
                    state = State::Ready;
                }
            }
        }

        if !std::matches!(state, State::Ready) {
            Err(Cmdline("missing argument".to_string()))
        } else {
            c.validate()?;
            Ok(c)
        }
    }

    pub fn update(&mut self, newer: &Self) -> Result<(), Error> {
        newer.use_omp.map(|x| self.use_omp.insert(x));
        newer.use_gpu.map(|x| self.use_gpu.insert(x));
        newer.device.map(|x| self.device.insert(x));
        self.upsample = newer.upsample;
        // newer.setup.as_ref().map(|x| self.setup.insert(x.to_string()));
        // newer.resolution.map(|x| self.resolution.insert(x));
        newer.fold.map(|x| self.fold.insert(x));
        newer.checkpoint_interval.map(|x| self.checkpoint_interval.insert(x));
        newer.checkpoint_logspace.map(|x| self.checkpoint_logspace.insert(x));
        newer.outdir.as_ref().map(|x| self.outdir.insert(x.to_string()));
        newer.end_time.map(|x| self.end_time.insert(x));
        newer.rk_order.map(|x| self.rk_order.insert(x));
        newer.cfl_number.map(|x| self.cfl_number.insert(x));
        newer.recompute_timestep.as_ref().map(|x| self.recompute_timestep.insert(x.to_string()));

        self.validate()
    }

    pub fn validate(&self) -> Result<(), Error> {
        use Error::*;

        if self.use_omp() && !crate::compiled_with_omp() {
            Err(CompiledWithoutOpenMP)
        } else if self.use_gpu() && !crate::compiled_with_gpu() {
            Err(CompiledWithoutGpu)
        } else if self.use_omp() && self.use_gpu() {
            Err(Cmdline(
                "--use-omp (-p) and --use-gpu (-g) are mutually exclusive".to_string(),
            ))
        } else if !(1..=3).contains(&self.rk_order()) {
            Err(Cmdline("rk-order must be 1, 2, or 3".into()))
        } else if self.checkpoint_interval() <= 0.0 {
            Err(Cmdline(
                "checkpoint interval --checkpoint (-c) must be >0".to_string(),
            ))
        } else if ![None, Some("iter"), Some("fold")].contains(&self.recompute_timestep.as_deref()) {
            Err(Cmdline(
                "invalid mode for --timestep, expected (iter|fold)".to_owned(),
            ))
        } else {
            Ok(())
        }
    }

    pub fn use_omp(&self) -> bool {
        self.use_omp.unwrap_or(false)
    }

    pub fn use_gpu(&self) -> bool {
        self.use_gpu.unwrap_or(false)
    }

    pub fn checkpoint_interval(&self) -> f64 {
        self.checkpoint_interval.unwrap_or(1.0)
    }

    pub fn rk_order(&self) -> usize {
        self.rk_order.unwrap_or(1)
    }

    pub fn fold(&self) -> usize {
        self.fold.unwrap_or(10)
    }

    pub fn cfl_number(&self) -> f64 {
        self.cfl_number.unwrap_or(0.2)
    }

    pub fn execution_mode(&self) -> ExecutionMode {
        if self.use_gpu() {
            ExecutionMode::GPU
        } else if self.use_omp() {
            ExecutionMode::OMP
        } else {
            ExecutionMode::CPU
        }
    }

    pub fn recompute_dt_each_iteration(&self) -> bool {
        match self.recompute_timestep.as_deref() {
            None => true,
            Some("iter") => true,
            Some("fold") => false,
            _ => panic!(),
        }
    }

    pub fn checkpoint_rule(&self, setup: &dyn Setup) -> Recurrence {
        if self.checkpoint_logspace.unwrap_or(false) {
            Recurrence::Log(self.checkpoint_interval())
        } else {
            Recurrence::Linear(self.checkpoint_interval() * setup.unit_time())
        }
    }

    pub fn simulation_end_time(&self, setup: &dyn Setup) -> f64 {
        self.end_time
            .or_else(|| setup.end_time())
            .map(|t| t * setup.unit_time())
            .unwrap_or(f64::MAX)
    }

    pub fn output_directory(&self, restart_file: &Option<String>) -> String {
        self.outdir
            .clone()
            .or_else(|| {
                restart_file
                    .as_deref()
                    .and_then(crate::parse::parent_dir)
                    .map(String::from)
            })
            .unwrap_or_else(|| String::from("."))
    }
}

impl Default for CommandLine {
    fn default() -> Self {
        Self {
            use_omp: None,
            use_gpu: None,
            device: None,
            upsample: None,
            resolution: None,
            fold: None,
            checkpoint_interval: None,
            checkpoint_logspace: None,
            setup: None,
            outdir: None,
            end_time: None,
            rk_order: None,
            cfl_number: None,
            recompute_timestep: None,
        }
    }
}
