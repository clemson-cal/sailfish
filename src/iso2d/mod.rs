use crate::sailfish::{BufferZone, EquationOfState, ExecutionMode, Mesh, PointMass, Solve};

extern "C" {
    fn iso2d_primitive_to_conserved(
        mesh: Mesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn iso2d_advance_rk(
        mesh: Mesh,
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
        mesh: Mesh,
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
pub fn advance<M: Fn(f64) -> Vec<PointMass>>(
    solver: &mut Box<dyn Solve>,
    eos: EquationOfState,
    buffer: BufferZone,
    masses: M,
    nu: f64,
    rk_order: u32,
    time: f64,
    dt: f64,
    velocity_ceiling: f64,
) {
    solver.primitive_to_conserved();

    match rk_order {
        1 => {
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.0 * dt), 0.0, dt, velocity_ceiling);
        }
        2 => {
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.0 * dt), 0.0, dt, velocity_ceiling);
            solver.advance_rk(nu, eos, buffer, &masses(time + 1.0 * dt), 0.5, dt, velocity_ceiling);
        }
        3 => {
            // t1 = a1 * tn + (1 - a1) * (tn + dt) =     tn +     (      dt) = tn +     dt [a1 = 0]
            // t2 = a2 * tn + (1 - a2) * (t1 + dt) = 3/4 tn + 1/4 (tn + 2dt) = tn + 1/2 dt [a2 = 3/4]
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.0 * dt), 0. / 1., dt, velocity_ceiling);
            solver.advance_rk(nu, eos, buffer, &masses(time + 1.0 * dt), 3. / 4., dt, velocity_ceiling);
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.5 * dt), 1. / 3., dt, velocity_ceiling);
        }
        _ => {
            panic!("invalid RK order")
        }
    }
}

pub mod cpu {
    use super::*;
    pub struct Solver {
        mesh: Mesh,
        primitive1: Vec<f64>,
        primitive2: Vec<f64>,
        conserved0: Vec<f64>,
        pub(super) mode: ExecutionMode,
    }

    impl Solver {
        pub fn new(mesh: Mesh, primitive: Vec<f64>) -> Self {
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
            nu: f64,
            eos: EquationOfState,
            buffer: BufferZone,
            masses: &[PointMass],
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
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
        fn max_wavespeed(&self, eos: EquationOfState, masses: &[PointMass]) -> f64 {
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
        pub fn new(mesh: Mesh, primitive: Vec<f64>) -> Self {
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
            nu: f64,
            eos: EquationOfState,
            buffer: BufferZone,
            masses: &[PointMass],
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            self.0.advance_rk(nu, eos, buffer, masses, a, dt, velocity_ceiling)
        }
        fn max_wavespeed(&self, eos: EquationOfState, masses: &[PointMass]) -> f64 {
            self.0.max_wavespeed(eos, masses)
        }
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use gpu_mem::DeviceVec;

    pub struct Solver {
        mesh: Mesh,
        primitive1: DeviceVec<f64>,
        primitive2: DeviceVec<f64>,
        conserved0: DeviceVec<f64>,
        wavespeeds: DeviceVec<f64>,
    }

    impl Solver {
        pub fn new(mesh: Mesh, primitive: Vec<f64>) -> Self {
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
            nu: f64,
            eos: EquationOfState,
            buffer: BufferZone,
            masses: &[PointMass],
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            let masses = DeviceVec::from(masses);

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
        fn max_wavespeed(&self, eos: EquationOfState, masses: &[PointMass]) -> f64 {
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
