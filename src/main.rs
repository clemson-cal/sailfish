pub mod cmdline;
pub mod physics;
pub mod error;

use kepler_two_body::{OrbitalElements, OrbitalState};
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

fn point_masses(state: OrbitalState, rate: f64, radius: f64) -> [PointMass; 2] {
    let OrbitalState(mass0, mass1) = state;
    let mass0 = PointMass {
        x: mass0.position_x(),
        y: mass0.position_y(),
        mass: mass0.mass(),
        rate,
        radius,
    };
    let mass1 = PointMass {
        x: mass1.position_x(),
        y: mass1.position_y(),
        mass: mass1.mass(),
        rate,
        radius,
    };
    [mass0, mass1]
}

fn build_solver(mesh: Mesh, use_omp: bool) -> Result<Box<dyn Solve>, error::Error> {
    if use_omp {
        #[cfg(feature="omp")]
        {
            Ok(Box::new(iso2d_omp::Solver::new(mesh)))
        }

        #[cfg(not(feature="omp"))]
        {
            Err(error::Error::CompiledWithoutOpenMP)
        }
    } else {
        Ok(Box::new(iso2d_cpu::Solver::new(mesh)))
    }
}

fn run() -> Result<(), error::Error> {
    use physics::f64::*;

    let cmdline = cmdline::parse_command_line()?;
    let mesh = Mesh {
        ni: cmdline.resolution,
        nj: cmdline.resolution,
        x0: -8.0,
        x1: 8.0,
        y0: -8.0,
        y1: 8.0,
    };
    let si = 3 * mesh.nj();
    let sj = 3;

    let sink_radius: f64 = 0.025;
    let sink_rate: f64 = 40.0;
    let mut primitive: Vec<f64> = vec![0.0; 3 * mesh.num_total_zones()];
    let mut solver = build_solver(mesh.clone(), cmdline.use_omp)?;

    let a: f64 = 1.0;
    let m: f64 = 1.0;
    let q: f64 = 1.0;
    let e: f64 = 0.0;
    let binary = OrbitalElements(a, m, q, e);

    for i in 0..mesh.ni() {
        for j in 0..mesh.nj() {
            let x = mesh.x0 + (i as f64 + 0.5) * mesh.dx();
            let y = mesh.y0 + (j as f64 + 0.5) * mesh.dy();
            let r = (x * x + y * y).sqrt();
            let rs = (x * x + y * y + sink_radius.powf(2.0)).sqrt();
            let phi_hat_x = -y / r;
            let phi_hat_y = x / r;
            let d = 1.0;
            let u = phi_hat_x / rs.sqrt();
            let v = phi_hat_y / rs.sqrt();
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
    let v_max = 1.0 / sink_radius.sqrt();
    let cfl = 0.2;
    let dt = mesh.dx().min(mesh.dy()) / v_max * cfl;
    let fold = cmdline.fold;

    let eos = EquationOfState::LocallyIsothermal { mach_number: 10.0 };
    // let eos = EquationOfState::Isothermal { sound_speed: 0.01 };
    // let buffer = BufferZone::None;
    let buffer = BufferZone::Keplerian {
        central_mass: 1.0,
        surface_density: 1.0,
        driving_rate: 1e3,
        outer_radius: 8.0,
        onset_width: 1.0,
    };

    while time < 600.0 {
        if time >= next_output_time {
            do_output(&solver.primitive(), output_number);
            output_number += 1;
            next_output_time += output_cadence;
        }
        let start = std::time::Instant::now();

        for _ in 0..fold {
            let masses = point_masses(binary.orbital_state_from_time(time), sink_rate, sink_radius);
            solver.compute_fluxes(eos, &masses);
            solver.advance(eos, buffer, &masses, dt);

            time += dt;
            iteration += 1;
        }
        let seconds = start.elapsed().as_secs_f64();
        let mzps = (mesh.ni() * mesh.nj()) as f64 / 1e6 / seconds * fold as f64;
        println!("[{}] t={:.3} Mzps={:.3}", iteration, time, mzps);
    }
    do_output(&solver.primitive(), output_number);
    Ok(())
}

fn main() {
    match run() {
        Ok(_) => {},
        Err(e) => {
            let message = format!("{}", e);
            print!("{}", message);
            if !message.ends_with("\n") {
                println!()
            }
        }        
    }
}
