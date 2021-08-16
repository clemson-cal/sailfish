use crate::cmdline::CommandLine;
use crate::error;
use crate::{Mesh, Patch, PointMass, Setup};
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::Write;

#[derive(Clone, Copy)]
pub enum Recurrence {
    Linear(f64),
    Log(f64),
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RecurringTask {
    pub number: u64,
    pub last_time: Option<f64>,
}

impl Default for RecurringTask {
    fn default() -> Self {
        Self::new()
    }
}

impl RecurringTask {
    pub fn new() -> Self {
        Self {
            number: 0,
            last_time: None,
        }
    }
    pub fn next(&mut self, current_time: f64, recurrence: Recurrence) {
        self.last_time = Some(self.next_time(current_time, recurrence));
        self.number += 1;
    }
    pub fn next_time(&self, current_time: f64, recurrence: Recurrence) -> f64 {
        if let Some(last_time) = self.last_time {
            match recurrence {
                Recurrence::Linear(interval) => last_time + interval,
                Recurrence::Log(multiplier) => last_time * (1.0 + multiplier),
            }
        } else {
            current_time
        }
    }
    pub fn is_due(&self, current_time: f64, recurrence: Recurrence) -> bool {
        current_time >= self.next_time(current_time, recurrence)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub command_line: CommandLine,
    pub restart_file: Option<String>,
    pub mesh: Mesh,
    pub setup_name: String,
    pub parameters: String,
    pub primitive: Vec<f64>,
    pub primitive_patches: Vec<Patch>,
    pub time: f64,
    #[serde(default)]
    pub masses: Vec<PointMass>,
    pub iteration: u64,
    pub checkpoint: RecurringTask,
}

impl State {
    pub fn from_checkpoint(
        filename: &str,
        new_parameters: &str,
        command_line: &CommandLine,
    ) -> Result<State, error::Error> {
        println!("read {}", filename);

        let mut f = File::open(filename).map_err(error::Error::IOError)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).map_err(error::Error::IOError)?;

        let mut state: State = rmp_serde::from_read_ref(&bytes)
            .map_err(|e| error::Error::InvalidCheckpoint(format!("{}", e)))?;

        if !state.parameters.is_empty() && !new_parameters.is_empty() {
            state.parameters += ":";
        }
        state.parameters += new_parameters;
        state.restart_file = Some(filename.to_string());
        state.command_line.update(&command_line)?;

        Ok(state)
    }

    pub fn set_primitive(&mut self, primitive: Vec<f64>) {
        assert!(
            primitive.len() == self.primitive.len(),
            "new and old primitive array sizes must match"
        );
        self.primitive = primitive;
    }

    pub fn write_checkpoint(
        &mut self,
        setup: &dyn Setup,
        outdir: &str,
    ) -> Result<(), error::Error> {
        let filename = format!("{}/chkpt.{:04}.sf", outdir, self.checkpoint.number);
        println!("write {}", filename);

        self.checkpoint
            .next(self.time, self.command_line.checkpoint_rule(setup));

        self.masses = setup.masses(self.time).to_vec();
        create_dir_all(outdir).map_err(error::Error::IOError)?;
        let bytes = rmp_serde::to_vec_named(self).unwrap();
        let mut file = File::create(&filename).map_err(error::Error::IOError)?;
        file.write_all(&bytes).map_err(error::Error::IOError)?;
        Ok(())
    }

    pub fn upsample(mut self) -> Self {
        let mut mesh = match self.mesh {
            Mesh::Structured(ref mut mesh) => mesh,
            _ => panic!("can only upsample structured mesh"),
        };

        for patch in &mut self.primitive_patches {
            patch.upsample_mut()
        }
        mesh.ni *= 2;
        mesh.nj *= 2;
        mesh.dx *= 0.5;
        mesh.dy *= 0.5;
        self
    }
}
