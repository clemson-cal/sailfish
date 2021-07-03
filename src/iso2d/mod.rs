use crate::sailfish::{BufferZone, EquationOfState, ExecutionMode, Mesh, PointMass, Solve};

extern "C" {
    pub fn iso2d_primitive_to_conserved(
        mesh: Mesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn iso2d_advance_rk(
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
        mode: ExecutionMode,
    );

    pub fn iso2d_wavespeed(
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
pub fn advance<M: Fn(f64) -> Vec<PointMass>>(
    solver: &mut Box<dyn Solve>,
    eos: EquationOfState,
    buffer: BufferZone,
    masses: M,
    nu: f64,
    rk_order: u32,
    time: f64,
    dt: f64)
{
    solver.primitive_to_conserved();

    match rk_order {
        1 => {
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.0 * dt), 0.0, dt);
        }
        2 => {
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.0 * dt), 0.0, dt);
            solver.advance_rk(nu, eos, buffer, &masses(time + 1.0 * dt), 0.5, dt);
        }
        3 => {
            // t1 = a1 * tn + (1 - a1) * (tn + dt) =     tn +     (      dt) = tn +     dt [a1 = 0]
            // t2 = a2 * tn + (1 - a2) * (t1 + dt) = 3/4 tn + 1/4 (tn + 2dt) = tn + 1/2 dt [a2 = 3/4]
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.0 * dt), 0. / 1., dt);
            solver.advance_rk(nu, eos, buffer, &masses(time + 1.0 * dt), 3. / 4., dt);
            solver.advance_rk(nu, eos, buffer, &masses(time + 0.5 * dt), 1. / 3., dt);
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
    }

    impl Solver {
        pub fn new(mesh: Mesh, primitive: Vec<f64>) -> Self {
            assert_eq!(primitive.len(), (mesh.ni as usize + 4) * (mesh.nj as usize + 4) * 3);
            Self {
                mesh,
                primitive1: primitive.clone(),
                primitive2: primitive.clone(),
                conserved0: vec![0.0; mesh.num_total_zones() * 3],
            }
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            self.primitive1.to_vec()
        }
        fn primitive_to_conserved(&mut self) {
            unsafe {
                iso2d_primitive_to_conserved(
                    self.mesh,
                    self.primitive1.as_ptr(),
                    self.conserved0.as_mut_ptr(),
                    ExecutionMode::CPU,
                );
            }
        }
        fn advance_rk(&mut self, nu: f64, eos: EquationOfState, buffer: BufferZone, masses: &[PointMass], a: f64, dt: f64) {
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
                    ExecutionMode::CPU,
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
                    ExecutionMode::CPU,
                )
            };
            wavespeeds
                .iter()
                .cloned()
                .fold(0.0, f64::max)
        }
    }
}
