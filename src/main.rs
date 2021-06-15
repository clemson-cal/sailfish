pub mod cmdline;
pub mod error;
pub mod physics;
pub mod setup;

use crate::setup::Setup;
use physics::f64::*;
use std::io::Write;

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

fn build_solver(mesh: Mesh, use_omp: bool) -> Result<Box<dyn Solve>, error::Error> {
    if use_omp {
        #[cfg(feature = "omp")]
        {
            Ok(Box::new(iso2d_omp::Solver::new(mesh)))
        }
        #[cfg(not(feature = "omp"))]
        {
            Err(error::Error::CompiledWithoutOpenMP)
        }
    } else {
        Ok(Box::new(iso2d_cpu::Solver::new(mesh)))
    }
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
    use physics::f64::*;
    let setup = setup::Shocktube {};
    let cmdline = cmdline::parse_command_line()?;
    let mesh = Mesh {
        ni: cmdline.resolution,
        nj: cmdline.resolution,
        x0: -8.0,
        x1: 8.0,
        y0: -8.0,
        y1: 8.0,
    };
    let mut solver = build_solver(mesh.clone(), cmdline.use_omp)?;

    let [si, sj] = mesh.strides();
    let mut primitive: Vec<f64> = vec![0.0; 3 * mesh.num_total_zones()];
    for i in 0..mesh.ni() {
        for j in 0..mesh.nj() {
            let x = mesh.x0 + (i as f64 + 0.5) * mesh.dx();
            let y = mesh.y0 + (j as f64 + 0.5) * mesh.dy();
            let n = i * si + j * sj;
            setup.initial_primitive(x, y, &mut primitive[n..n + 3]);
        }
    }
    solver.set_primitive(&primitive);

    let mut time = 0.0;
    let mut iteration = 0;
    let mut output_number = 0;
    let mut next_output_time = time;
    let mut mzps_log = vec![];

    let v_max = setup.max_signal_speed().unwrap();
    let cfl = cmdline.cfl_number;
    let fold = cmdline.fold;
    let checkpoint_interval = cmdline.checkpoint_interval;
    let dt = mesh.dx().min(mesh.dy()) / v_max * cfl;

    let eos = setup.equation_of_state();
    let buffer = setup.buffer_zone();

    println!("omp enabled: {}", cfg!(feature = "omp"));
    println!("cuda enabled: {}", cfg!(feature = "cuda"));

    while time < cmdline.end_time {
        if time >= next_output_time {
            do_output(&solver.primitive(), output_number);
            output_number += 1;
            next_output_time += checkpoint_interval;
        }

        let elapsed = time_exec(|| {
            for _ in 0..fold {
                let masses = setup.particles(time);

                if cmdline.precompute_flux {
                    solver.compute_fluxes(eos, &masses);
                }
                solver.advance(eos, buffer, &masses, dt);

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

    println!(
        "average perf: {:.3} Mzps",
        mzps_log.iter().fold(0.0, |a, b| a + b) / mzps_log.len() as f64
    );
    Ok(())
}

fn main() {
    match run() {
        Ok(_) => {}
        Err(e) => print!("{}", e),
    }
}
