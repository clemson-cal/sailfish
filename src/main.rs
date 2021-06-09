pub mod physics;

use physics::*;
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

fn main() {
    use physics::f64::*;

    let mesh = Mesh {
        ni: 512,
        nj: 512,
        x0: -5.0,
        x1:  5.0,
        y0: -5.0,
        y1:  5.0,
    };
    let si = 3 * mesh.nj();
    let sj = 3;

    let r_soft: f64 = 0.1;
    let mut primitive: Vec<f64> = vec![0.0; 3 * mesh.num_total_zones()];
    let mut solver = iso2d_cpu::Solver::new(mesh.clone());

    for i in 0..mesh.ni() {
        for j in 0..mesh.nj() {
            let x = mesh.x0 + (i as f64 + 0.5) * mesh.dx();
            let y = mesh.y0 + (j as f64 + 0.5) * mesh.dy();
            let r = (x * x + y * y + r_soft.powf(2.0)).sqrt();
            let phi_hat_x = -y / r;
            let phi_hat_y =  x / r;
            let d = 1.0;
            let u = phi_hat_x / r.sqrt();
            let v = phi_hat_y / r.sqrt();
            primitive[i * si + j * sj + 0] = d;
            primitive[i * si + j * sj + 1] = u;
            primitive[i * si + j * sj + 2] = v;
        }
    }
    solver.set_primitive(&primitive);

    let mut time = 0.0;
    let mut iteration = 0;
    let mut output_number = 0;
    let mut next_output_time = time;
    let output_cadence = 1.0;
    let v_max = 1.0 / r_soft.sqrt();
    let cfl = 0.2;
    let dt = mesh.dx().min(mesh.dy()) / v_max * cfl;

    let mass = PointMass {
        x: 0.0,
        y: 0.0,
        mass: 1.0,
        rate: 40.0,
        radius: r_soft,
    };

    let masses = vec![mass];
    let eos = EquationOfState::LocallyIsothermal { mach_number: 10.0 };
    // let buffer = BufferZone::None;
    let buffer = BufferZone::Keplerian {
        central_mass: 1.0,
        surface_density: 1.0,
        driving_rate: 1e3,
        onset_radius: 5.0,
        onset_width: 1.0,
    };

    while time < 50.0 {
        if time >= next_output_time {
            do_output(&solver.primitive(), output_number);
            output_number += 1;
            next_output_time += output_cadence;
        }

        let start = std::time::Instant::now();
        solver.advance_cons(eos, buffer, &masses, dt);
        time += dt;
        iteration += 1;
        let seconds = start.elapsed().as_secs_f64();
        let mzps = (mesh.ni() * mesh.nj()) as f64 / 1e6 / seconds;
        println!("[{}] t={:.3} Mzps={:.3}", iteration, time, mzps);
    }

    do_output(&solver.primitive(), output_number);
}
