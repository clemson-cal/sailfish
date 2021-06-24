use crate::setup::Setup;
use crate::physics::ExecutionMode;
use std::io::Write;

pub mod cmdline;
pub mod error;
pub mod physics;
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

    use solver::host;

    let _patch = host::Patch::from_fn([0, 0], [128, 128], |_, _| { [0.0] });

    let cmdline = cmdline::parse_command_line()?;
    let mesh = physics::Mesh {
        x0: -1.0,
        y0: -1.0,
        ni: cmdline.resolution,
        nj: cmdline.resolution,
        dx: 2.0 / cmdline.resolution as f64,
        dy: 2.0 / cmdline.resolution as f64,
    };
    let mode = match (cmdline.use_omp, cmdline.use_gpu) {
        (false, false) => ExecutionMode::CPU,
        (true, false) => ExecutionMode::OMP,
        (_, true) => ExecutionMode::GPU,
    };

    let mut solver = physics::Solver::new(mesh.clone(), mode);
    let setup = setup::Explosion {};
    let [si, sj] = mesh.strides();
    let mut primitive: Vec<f64> = vec![0.0; 3 * mesh.num_total_zones()];
    for i in 0..mesh.ni() {
        for j in 0..mesh.nj() {
            let x = mesh.x0 + (i as f64 + 0.5) * mesh.dx;
            let y = mesh.y0 + (j as f64 + 0.5) * mesh.dy;
            let n = i * si + j * sj;
            setup.initial_primitive(x, y, &mut primitive[n..n + 3]);
        }
    }
    solver.set_primitive(&primitive);

    let v_max = 1.0;
    let rk_order = cmdline.rk_order;
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
                solver.advance(rk_order, dt);
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
