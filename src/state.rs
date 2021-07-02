use crate::error;
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::Write;

#[derive(Clone, Serialize, Deserialize)]
pub struct RecurringTask {
    pub number: u64,
    pub last_time: Option<f64>,
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
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub setup_name: String,
    pub parameters: String,
    pub primitive: Vec<f64>,
    pub time: f64,
    pub iteration: u64,
    pub checkpoint: RecurringTask,
}

impl State {
    pub fn from_checkpoint(filename: &str, new_parameters: &str) -> Result<State, error::Error> {
        let mut f = File::open(filename).map_err(error::Error::IOError)?;

        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).map_err(error::Error::IOError)?;

        let mut state: State = rmp_serde::from_read_ref(&bytes)
            .map_err(|e| error::Error::InvalidCheckpoint(format!("{}", e)))?;

        state.parameters += ":";
        state.parameters += new_parameters;

        println!("read {}", filename);
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
        file.write_all(&bytes).unwrap();
        println!("write {}", filename);
        Ok(())
    }
}
