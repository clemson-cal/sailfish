use crate::Setup;
use crate::sailfish::{BufferZone, EquationOfState, ExecutionMode, StructuredMesh, PointMass, Solve};
use cfg_if::cfg_if;

extern "C" {
    fn iso2d_primitive_to_conserved(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn iso2d_advance_rk(
        mesh: StructuredMesh,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        eos: EquationOfState,
        buffer: BufferZone,
        masses: *const PointMass,
        num_masses: i32,
        nu: f64,
        a: f64,
        dt: f64,
        velocity_ceiling: f64,
        mode: ExecutionMode,
    );

    fn iso2d_wavespeed(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        eos: EquationOfState,
        masses: *const PointMass,
        num_masses: i32,
        mode: ExecutionMode,
    );
}

/// Primitive variable array in a solver using first, second, or third-order
/// Runge-Kutta time stepping.
#[allow(clippy::too_many_arguments)]
pub fn advance(
    solver: &mut Box<dyn Solve>,
    setup: &Box<dyn Setup>,
    rk_order: u32,
    time: f64,
    dt: f64,
    velocity_ceiling: f64,
) {
    solver.primitive_to_conserved();
    match rk_order {
        1 => {
            solver.advance_rk(time, setup, 0.0, dt, velocity_ceiling);
        }
        2 => {
            solver.advance_rk(time + 0.0 * dt, setup, 0.0, dt, velocity_ceiling);
            solver.advance_rk(time + 1.0 * dt, setup, 0.5, dt, velocity_ceiling);
        }
        3 => {
            // t1 = a1 * tn + (1 - a1) * (tn + dt) =     tn +     (      dt) = tn +     dt [a1 = 0]
            // t2 = a2 * tn + (1 - a2) * (t1 + dt) = 3/4 tn + 1/4 (tn + 2dt) = tn + 1/2 dt [a2 = 3/4]
            solver.advance_rk(time + 0.0 * dt, setup, 0. / 1., dt, velocity_ceiling);
            solver.advance_rk(time + 1.0 * dt, setup, 3. / 4., dt, velocity_ceiling);
            solver.advance_rk(time + 0.5 * dt, setup, 1. / 3., dt, velocity_ceiling);
        }
        _ => {
            panic!("invalid RK order")
        }
    }
}

pub fn solver(mode: ExecutionMode, mesh: StructuredMesh, primitive: Vec<f64>) -> Box<dyn Solve> {
    match mode {
        ExecutionMode::CPU => Box::new(cpu::Solver::new(mesh, primitive)),
        ExecutionMode::OMP => {
            cfg_if! {
                if #[cfg(feature = "omp")] {
                    Box::new(omp::Solver::new(mesh, primitive))
                } else {
                    panic!()
                }
            }
        }
        ExecutionMode::GPU => {
            cfg_if! {
                if #[cfg(feature = "cuda")] {
                    Box::new(gpu::Solver::new(mesh, primitive))
                } else {
                    panic!()
                }
            }
        }
    }
}

pub mod cpu {
    use super::*;
    pub struct Solver {
        mesh: StructuredMesh,
        primitive1: Vec<f64>,
        primitive2: Vec<f64>,
        conserved0: Vec<f64>,
        pub(super) mode: ExecutionMode,
    }

    impl Solver {
        pub fn new(mesh: StructuredMesh, primitive: Vec<f64>) -> Self {
            assert_eq!(
                primitive.len(),
                (mesh.ni as usize + 4) * (mesh.nj as usize + 4) * 3
            );
            Self {
                mesh,
                primitive1: primitive.clone(),
                primitive2: primitive,
                conserved0: vec![0.0; mesh.num_total_zones() * 3],
                mode: ExecutionMode::CPU,
            }
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            self.primitive1.clone()
        }
        fn primitive_to_conserved(&mut self) {
            unsafe {
                iso2d_primitive_to_conserved(
                    self.mesh,
                    self.primitive1.as_ptr(),
                    self.conserved0.as_mut_ptr(),
                    self.mode,
                );
            }
        }
        fn advance_rk(
            &mut self,
            time: f64,
            setup: &Box<dyn Setup>,
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            let buffer = setup.buffer_zone();
            let eos = setup.equation_of_state();
            let nu = setup.viscosity().unwrap_or(0.0);
            let masses = setup.masses(time);
            unsafe {
                iso2d_advance_rk(
                    self.mesh,
                    self.conserved0.as_ptr(),
                    self.primitive1.as_ptr(),
                    self.primitive2.as_mut_ptr(),
                    eos,
                    buffer,
                    masses.as_ptr(),
                    masses.len() as i32,
                    nu,
                    a,
                    dt,
                    velocity_ceiling,
                    self.mode,
                )
            };
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, time: f64, setup: &Box<dyn Setup>) -> f64 {
            let eos = setup.equation_of_state();
            let masses = setup.masses(time);
            let mut wavespeeds = vec![0.0; self.mesh.num_total_zones()];
            unsafe {
                iso2d_wavespeed(
                    self.mesh,
                    self.primitive1.as_ptr(),
                    wavespeeds.as_mut_ptr(),
                    eos,
                    masses.as_ptr(),
                    masses.len() as i32,
                    self.mode,
                )
            };
            wavespeeds.iter().cloned().fold(0.0, f64::max)
        }
    }
}

pub mod omp {
    use super::*;
    pub struct Solver(cpu::Solver);

    impl Solver {
        pub fn new(mesh: StructuredMesh, primitive: Vec<f64>) -> Self {
            let mut solver = cpu::Solver::new(mesh, primitive);
            solver.mode = ExecutionMode::OMP;
            Self(solver)
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            self.0.primitive()
        }
        fn primitive_to_conserved(&mut self) {
            self.0.primitive_to_conserved()
        }
        fn advance_rk(
            &mut self,
            time: f64,
            setup: &Box<dyn Setup>,
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            self.0.advance_rk(time, setup, a, dt, velocity_ceiling)
        }
        fn max_wavespeed(&self, time: f64, setup: &Box<dyn Setup>) -> f64 {
            self.0.max_wavespeed(time, setup)
        }
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use gpu_mem::DeviceVec;

    pub struct Solver {
        mesh: StructuredMesh,
        primitive1: DeviceVec<f64>,
        primitive2: DeviceVec<f64>,
        conserved0: DeviceVec<f64>,
        wavespeeds: DeviceVec<f64>,
    }

    impl Solver {
        pub fn new(mesh: StructuredMesh, primitive: Vec<f64>) -> Self {
            assert_eq!(
                primitive.len(),
                (mesh.ni as usize + 4) * (mesh.nj as usize + 4) * 3
            );
            Self {
                mesh,
                primitive1: DeviceVec::from(&primitive),
                primitive2: DeviceVec::from(&primitive),
                conserved0: DeviceVec::from(&vec![0.0; mesh.num_total_zones() * 3]),
                wavespeeds: DeviceVec::from(&vec![0.0; mesh.num_total_zones()]),
            }
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            Vec::from(&self.primitive1)
        }
        fn primitive_to_conserved(&mut self) {
            unsafe {
                iso2d_primitive_to_conserved(
                    self.mesh,
                    self.primitive1.as_device_ptr(),
                    self.conserved0.as_mut_device_ptr(),
                    ExecutionMode::GPU,
                );
            }
        }
        fn advance_rk(
            &mut self,
            time: f64,
            setup: &Box<dyn Setup>,
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            let buffer = setup.buffer_zone();
            let eos = setup.equation_of_state();
            let nu = setup.viscosity().unwrap_or(0.0);
            let masses = DeviceVec::from(&setup.masses(time));

            unsafe {
                iso2d_advance_rk(
                    self.mesh,
                    self.conserved0.as_device_ptr(),
                    self.primitive1.as_device_ptr(),
                    self.primitive2.as_mut_device_ptr(),
                    eos,
                    buffer,
                    masses.as_device_ptr(),
                    masses.len() as i32,
                    nu,
                    a,
                    dt,
                    velocity_ceiling,
                    ExecutionMode::GPU,
                )
            };
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, time: f64, setup: &Box<dyn Setup>) -> f64 {
            let eos = setup.equation_of_state();
            let masses = setup.masses(t);
            use gpu_mem::Reduce;
            let masses = DeviceVec::from(masses);

            unsafe {
                iso2d_wavespeed(
                    self.mesh,
                    self.primitive1.as_device_ptr(),
                    self.wavespeeds.as_device_ptr() as *mut f64,
                    eos,
                    masses.as_device_ptr(),
                    masses.len() as i32,
                    ExecutionMode::GPU,
                )
            };
            self.wavespeeds.maximum().unwrap()
        }
    }
}
