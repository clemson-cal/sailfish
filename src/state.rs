use serde::{Serialize, Deserialize};
use std::io::Write;
use crate::error;

#[derive(Clone, Serialize, Deserialize)]
pub struct RecurringTask
{
    pub number: u64,
    pub next_time: f64,
}

impl RecurringTask {
    pub fn next(&mut self, interval: f64) {
        self.next_time += interval;
        self.number += 1;
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct State
{
    pub setup: String,
    pub primitive: Vec<f64>,
    pub time: f64,
    pub iteration: u64,
    pub checkpoint: RecurringTask,
}

pub fn write_checkpoint(state: &State, outdir: &str, output_number: u64) -> Result<(), error::Error> {

    // let mut bytes = Vec::new();
    // for x in &state.primitive {
    //     bytes.extend(x.to_le_bytes().iter());
    // }
    let bytes = rmp_serde::to_vec_named(&state).unwrap();

    std::fs::create_dir_all(outdir).map_err(error::Error::IOError)?;
    let filename = format!("{}/chkpt.{:04}.sf", outdir, output_number);
    let mut file = std::fs::File::create(&filename).unwrap();
    file.write_all(&bytes).unwrap();
    println!("write {}", filename);
    Ok(())
}
