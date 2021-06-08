pub mod physics;

use std::io::Write;
use physics::*;

fn do_output(primitive: Vec<f64>, output_number: usize) {
    let bytes: Vec<u8> = primitive
        .into_iter()
        .map(|x| x.to_le_bytes())
        .flatten()
        .collect();
    let filename = format!("sailfish.{:04}.bin", output_number);
    let mut file = std::fs::File::create(&filename).unwrap();
    file.write_all(&bytes).unwrap();
    println!("write {}", filename);
}

fn main() {
    let mesh = physics::f64::Mesh {
        ni: 512,
        nj: 512,
        x0: -0.5,
        x1: 0.5,
        y0: -0.5,
        y1: 0.5,
    };
    let si = 3 * mesh.nj();
    let sj = 3;

    let mut primitive: Vec<f64> = vec![0.0; 3 * mesh.num_total_zones()];
    let mut solver = iso2d_cpu_f64::Solver::new(mesh.clone());

    for i in 0..mesh.ni() {
        for j in 0..mesh.nj() {

            primitive[i * si + j * sj + 0] = 1.0;
            primitive[i * si + j * sj + 1] = 0.0;
            primitive[i * si + j * sj + 2] = 0.0;
        }
    }
    solver.set_primitive(&primitive);

    let mut time = 0.0;
    let mut iteration = 0;
    let mut output_number = 0;
    let mut next_output_time = time;
    let output_cadence = 0.01;
    let dt = mesh.dx().min(mesh.dy()) * 0.01;

    let mass1 = physics::f64::PointMass{
        x: 0.0,
        y: 0.0,
        mass: 1.0,
        rate: 1.0,
        radius: 0.05,
    };

    let masses = vec![mass1];
    let eos = physics::f64::EquationOfState::LocallyIsothermal{ mach_number: 10.0 };

    while time < 0.1 {

        if time >= next_output_time {
            do_output(solver.primitive(), output_number);
            output_number += 1;
            next_output_time += output_cadence;
        }

        let start = std::time::Instant::now();
        solver.advance_cons(eos, &masses, dt);
        time += dt;
        iteration += 1;
        let seconds = start.elapsed().as_secs_f64();
        let mzps = (mesh.ni() * mesh.nj()) as f64 / 1e6 / seconds;
        println!("[{}] t={:.3} Mzps={:.3}", iteration, time, mzps);
    }

    do_output(solver.primitive(), output_number);
}
