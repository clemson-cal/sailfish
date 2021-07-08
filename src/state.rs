use crate::cmdline::CommandLine;
use crate::error;
use crate::mesh;
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::Write;

#[derive(Clone, Serialize, Deserialize)]
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
        Self { number: 0, last_time: None }
    }
    pub fn next(&mut self, interval: f64) {
        if self.last_time.is_none() {
            self.last_time = Some(0.0);
        } else {
            self.last_time = Some(self.last_time.unwrap() + interval);
        }
        self.number += 1;
    }
    pub fn is_due(&self, time: f64, interval: f64) -> bool {
        self.last_time.map_or(true, |last_time| time >= last_time + interval)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub command_line: CommandLine,
    pub restart_file: Option<String>,
    pub mesh: mesh::Mesh,
    pub setup_name: String,
    pub parameters: String,
    pub primitive: Vec<f64>,
    pub time: f64,
    pub iteration: u64,
    pub checkpoint: RecurringTask,
}

impl State {
    pub fn from_checkpoint(filename: &str, new_parameters: &str) -> Result<State, error::Error> {
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

        Ok(state)
    }

    pub fn set_primitive(&mut self, primitive: Vec<f64>) {
        assert!(primitive.len() == self.primitive.len(),
            "new and old primitive array sizes must match");
        self.primitive = primitive;
    }

    pub fn write_checkpoint(
        &mut self,
        checkpoint_interval: f64,
        outdir: &str,
    ) -> Result<(), error::Error> {
        self.checkpoint.next(checkpoint_interval);
        create_dir_all(outdir).map_err(error::Error::IOError)?;
        let bytes = rmp_serde::to_vec_named(self).unwrap();
        let filename = format!("{}/chkpt.{:04}.sf", outdir, self.checkpoint.number - 1);
        let mut file = File::create(&filename).unwrap();
        println!("write {}", filename);
        file.write_all(&bytes).unwrap();
        Ok(())
    }

    pub fn upsample(mut self) -> Self {
        let mut mesh = match self.mesh {
            mesh::Mesh::Structured(ref mut mesh) => mesh,
            _ => panic!("can only upsample structured mesh")
        };
        let ni = mesh.ni;
        let nj = mesh.nj;
        let mi = 2 * ni;
        let mj = 2 * nj;
        let mut new_primitive = vec![0.0; (mi as usize + 4) * (mj as usize + 4) * 3];

        for i in -2..ni + 2 {
            for j in -2..nj + 2 {
                let i0 = 2 * i;
                let i1 = 2 * i + 1;
                let j0 = 2 * j;
                let j1 = 2 * j + 1;

                for q in 0..3 {
                    let p = self.primitive[(i + 2) as usize * (nj as usize + 4) * 3 + (j + 2) as usize * 3 + q];

                    if (-2..mi + 2).contains(&i0) && (-2..mj + 2).contains(&j0) {
                        new_primitive[(i0 + 2) as usize * (mj as usize + 4) * 3 + (j0 + 2) as usize * 3 + q] = p;
                    }
                    if (-2..mi + 2).contains(&i0) && (-2..mj + 2).contains(&j1) {
                        new_primitive[(i0 + 2) as usize * (mj as usize + 4) * 3 + (j1 + 2) as usize * 3 + q] = p;
                    }
                    if (-2..mi + 2).contains(&i1) && (-2..mj + 2).contains(&j0) {
                        new_primitive[(i1 + 2) as usize * (mj as usize + 4) * 3 + (j0 + 2) as usize * 3 + q] = p;
                    }
                    if (-2..mi + 2).contains(&i1) && (-2..mj + 2).contains(&j1) {
                        new_primitive[(i1 + 2) as usize * (mj as usize + 4) * 3 + (j1 + 2) as usize * 3 + q] = p;
                    }
                }
            }
        }
        self.primitive = new_primitive;
        mesh.ni *= 2;
        mesh.nj *= 2;
        mesh.dx *= 0.5;
        mesh.dy *= 0.5;
        self
    }
}
