use setup::Setup;
use std::io::Write;
use solver::{cpu, omp, Solve};

pub mod cmdline;
pub mod error;
pub mod setup;
pub mod solver;

fn do_output(primitive: &Vec<f64>, output_number: usize) {
    let mut bytes = Vec::new();
    for x in primitive {
        bytes.extend(x.to_le_bytes().iter());
    }
    let filename = format!("sailfish.{:04}.bin", output_number);
    let mut file = std::fs::File::create(&filename).unwrap();
    file.write_all(&bytes).unwrap();
    println!("write {}", filename);
}

fn time_exec<F>(mut f: F) -> std::time::Duration
where
    F: FnMut(),
{
    let start = std::time::Instant::now();
    f();
    start.elapsed()
}

fn run() -> Result<(), error::Error> {
    let cmdline = cmdline::parse_command_line()?;
    let mesh = solver::Mesh::centered_square(1.0, cmdline.resolution);
    let setup = setup::Explosion {};

    let primitive = setup.initial_primitive_vec(&mesh);
    let mut solver: Box<dyn Solve> = match (cmdline.use_omp, cmdline.use_gpu) {
        (false, false) => Box::new(cpu::Solver::new(mesh.clone(), primitive)),
        (true, false) => Box::new(omp::Solver::new(mesh.clone(), primitive)),
        (_, true) => todo!(),
    };

    let v_max = 1.0;
    let cfl = cmdline.cfl_number;
    let fold = cmdline.fold;
    let checkpoint_interval = cmdline.checkpoint_interval;
    let dt = f64::min(mesh.dx, mesh.dy) / v_max * cfl;

    let mut time = 0.0;
    let mut iteration = 0;
    let mut output_number = 0;
    let mut next_output_time = time;
    let mut mzps_log = vec![];

    while time < cmdline.end_time {
        if time >= next_output_time {
            do_output(&solver.primitive(), output_number);
            output_number += 1;
            next_output_time += checkpoint_interval;
        }

        let elapsed = time_exec(|| {
            for _ in 0..fold {
                solver.advance(1, dt);
                time += dt;
                iteration += 1;
            }
        });

        mzps_log.push((mesh.num_total_zones() * fold) as f64 / 1e6 / elapsed.as_secs_f64());
        println!(
            "[{}] t={:.3} Mzps={:.3}",
            iteration,
            time,
            mzps_log.last().unwrap()
        );
    }
    do_output(&solver.primitive(), output_number);
    Ok(())
}

fn main() {
    match run() {
        Ok(_) => {}
        Err(e) => print!("{}", e),
    }
}
