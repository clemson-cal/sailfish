use setup::Setup;
use std::io::Write;

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
    use solver::host;

    let cmdline = cmdline::parse_command_line()?;
    let mesh = solver::Mesh::centered_square(1.0, cmdline.resolution);
    let setup = setup::Explosion {};

    let mut primitive1 = host::Patch::from_fn([-2, -2], [mesh.ni() + 4, mesh.nj() + 4], |i, j| {
        let [x, y] = mesh.cell_coordinates(i, j);
        let mut p = [0.0; 3];
        setup.initial_primitive(x, y, &mut p);
        p
    });
    let mut primitive2 = primitive1.clone();
    let mut conserved = host::Patch::from_fn([0, 0], mesh.shape(), |_, _| [0.0, 0.0, 0.0]);

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
            do_output(&primitive1.to_vec(), output_number);
            output_number += 1;
            next_output_time += checkpoint_interval;
        }

        let elapsed = time_exec(|| {
            for _ in 0..fold {
                solver::iso2d::primitive_to_conserved_cpu(&primitive1, &mut conserved);
                solver::iso2d::advance_rk_cpu(&mesh, &conserved, &primitive1, &mut primitive2, 0.0, dt);
                std::mem::swap(&mut primitive1, &mut primitive2);
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
    do_output(&primitive1.to_vec(), output_number);
    Ok(())
}

fn main() {
    match run() {
        Ok(_) => {}
        Err(e) => print!("{}", e),
    }
}
