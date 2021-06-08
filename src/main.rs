use std::io::Write;
pub mod physics;

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
    let config = physics::Configuration {
        grid_dim: 1024,
        sink_rate: 8.0,
        sink_radius: 0.05,
        mach_number: 1.0,
        domain_radius: 0.5,
    };
    let si = 3 * config.nj();
    let sj = 3;

    let mut primitive: Vec<f64> = vec![0.0; (config.grid_dim as usize).pow(2) * 3];
    let mut solver = physics::iso2d_cpu_f64::Solver::new(config.clone());

    let dx = 1.0 / config.grid_dim as f64;

    for i in 0..config.ni() {
        for j in 0..config.nj() {
            let x = -0.5 + (i as f64 + 0.5) * dx;
            let y = -0.5 + (j as f64 + 0.5) * dx;

            if (x * x + y * y).sqrt() < 0.2 {
                primitive[i * si + j * sj + 0] = 1.0;
                primitive[i * si + j * sj + 1] = 0.0;
                primitive[i * si + j * sj + 2] = 0.0;
            } else {
                primitive[i * si + j * sj + 0] = 0.1;
                primitive[i * si + j * sj + 1] = 0.0;
                primitive[i * si + j * sj + 2] = 0.0;
            }
        }
    }
    solver.set_primitive(&primitive);
    drop(primitive);

    let mut t = 0.0;
    let mut iteration = 0;
    let mut output_number = 0;
    let output_cadence = 0.1;
    let dt = dx * 0.1;

    while t < 0.4 {

        if t >= output_cadence * output_number as f64 {
            do_output(solver.primitive(), output_number);
            output_number += 1;
        }

        let start = std::time::Instant::now();
        solver.advance_cons(dt);
        t += dt;
        iteration += 1;
        let seconds = start.elapsed().as_secs_f64();
        let mzps = (config.ni() * config.nj()) as f64 / 1e6 / seconds;
        println!("[{}] t={:.3} Mzps={:.3}", iteration, t, mzps);
    }
}
