pub mod physics;

use kepler_two_body::{OrbitalElements, OrbitalState};
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

fn point_masses(state: OrbitalState, rate: f64, radius: f64) -> [f64::PointMass; 2] {
    let OrbitalState(mass0, mass1) = state;
    let mass0 = f64::PointMass {
        x: mass0.position_x(),
        y: mass0.position_y(),
        mass: mass0.mass(),
        rate,
        radius,
    };
    let mass1 = f64::PointMass {
        x: mass1.position_x(),
        y: mass1.position_y(),
        mass: mass1.mass(),
        rate,
        radius,
    };
    [mass0, mass1]
}

fn run(cmdline: CommandLine) {
    use physics::f64::*;

    println!("{:?}", cmdline);

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

    let sink_radius: f64 = 0.1;
    let sink_rate: f64 = 10.0;
    let mut primitive: Vec<f64> = vec![0.0; 3 * mesh.num_total_zones()];
    let mut solver: Box<dyn Solve> = if cmdline.no_omp {
        Box::new(iso2d_cpu::Solver::new(mesh.clone()))
    } else {
        Box::new(iso2d_omp::Solver::new(mesh.clone()))
    };

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

    while time < 20.0 {
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
}

#[derive(Debug)]
struct CommandLine {
    no_omp: bool,
    resolution: u64,
    fold: u32,
}

fn main() {
    let mut c = CommandLine {
        no_omp: false,
        resolution: 1024,
        fold: 100,
    };

    enum State {
        Ready,
        GridResolution,
        Fold,
    }
    let mut state = State::Ready;

    for arg in std::env::args()
        .skip(1)
        .map(|arg| arg.split("=").map(str::to_string).collect::<Vec<_>>())
        .flatten()
    {
        match state {
            State::Ready => match arg.as_str() {
                "-h" | "--help" => {
                    println!("   -h  | --help          display this help message");
                    println!("   --version             print the code version number");
                    println!("   -no-omp | --no-omp    disable running with OpenMP");
                    println!("   -n | --resolution     grid resolution [1024]");
                    println!("   -f | --fold           number of iterations between messages");
                    return;
                }
                "--version" => {
                    println!("sailfish 0.1.0 {}", git_version::git_version!());
                    return;
                }
                "-no-omp" | "--no-omp" => {
                    c.no_omp = true;
                }
                "-n" | "--res" => {
                    state = State::GridResolution;
                }
                "-f" | "--fold" => {
                    state = State::Fold;
                }
                _ => {
                    eprintln!("unrecognized option {}", arg);
                    return;
                }
            },
            State::GridResolution => match arg.parse() {
                Ok(n) => {
                    c.resolution = n;
                    state = State::Ready;
                }
                Err(e) => {
                    eprintln!("-n | --resolution {}: {}", arg, e);
                    return;
                }
            },
            State::Fold => match arg.parse() {
                Ok(f) => {
                    c.fold = f;
                    state = State::Ready;
                }
                Err(e) => {
                    eprintln!("-f | --fold {}: {}", arg, e);
                    return;
                }
            },
        }
    }

    if !std::matches!(state, State::Ready) {
        eprintln!("missing argument");
        return;
    }
    run(c)
}
