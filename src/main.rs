pub mod physics;

fn main() {
    let config = physics::Configuration {
        grid_dim: 1024,
        sink_rate: 8.0,
        sink_radius: 0.05,
        mach_number: 10.0,
        domain_radius: 24.0,
    };
    let si = 3 * config.nj();
    let sj = 3;

    let mut primitive: Vec<f64> = vec![0.0; (config.grid_dim as usize).pow(2) * 3];
    let mut update = physics::iso2d_cpu_f64::Solver::new(config.clone());

    for i in 0..config.ni() {
        for j in 0..config.nj() {
            primitive[i * si + j * sj + 0] = 1.0;
            primitive[i * si + j * sj + 1] = 0.0;
            primitive[i * si + j * sj + 2] = 1.0;
        }
    }
    update.set_primitive(&primitive);

    let mut t = 0.0;
    let mut iteration = 0;
    let dt = 0.0001;

    while t < 0.01 {
        let start = std::time::Instant::now();
        update.advance_cons(dt);
        t += dt;
        iteration += 1;
        let seconds = start.elapsed().as_secs_f64();
        let mzps = (config.ni() * config.nj()) as f64 / 1e6 / seconds;
        println!("[{}] t={:.3} Mzps={:.3}", iteration, t, mzps);
    }
}
