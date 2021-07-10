use crate::sailfish::{ExecutionMode, Solve};

extern "C" {
    fn euler1d_primitive_to_conserved(
        num_zones: i32,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode);

    fn euler1d_advance_rk(
        num_zones: i32,
        face_positions_ptr: *const f64,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        a: f64,
        dt: f64,
        mode: ExecutionMode);

    fn euler1d_wavespeed(
        num_zones: i32,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        mode: ExecutionMode);
}

pub mod cpu {
    use crate::Setup;
    use super::*;

    pub struct Solver {
        face_positions: Vec<f64>,
        primitive1: Vec<f64>,
        primitive2: Vec<f64>,
        conserved0: Vec<f64>,
        pub(super) mode: ExecutionMode,
    }

    impl Solver {
        pub fn new(face_positions: Vec<f64>, primitive: Vec<f64>) -> Self {
            let num_zones = face_positions.len() - 1;
            assert_eq!(
                primitive.len(),
                num_zones * 3
            );
            Self {
                face_positions,
                primitive1: primitive.clone(),
                primitive2: primitive,
                conserved0: vec![0.0; num_zones * 3],
                mode: ExecutionMode::CPU,
            }
        }

        fn num_zones(&self) -> usize {
            self.face_positions.len() - 1
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            self.primitive1.clone()
        }
        fn primitive_to_conserved(&mut self) {
            unsafe {
                euler1d_primitive_to_conserved(
                    self.num_zones() as i32,
                    self.primitive1.as_ptr(),
                    self.conserved0.as_mut_ptr(),
                    self.mode,
                );
            }
        }
        fn advance_rk(
            &mut self,
            _setup: &dyn Setup,
            _time: f64,
            a: f64,
            dt: f64,
            _velocity_ceiling: f64,
        ) {
            unsafe {
                euler1d_advance_rk(
                    self.num_zones() as i32,
                    self.face_positions.as_ptr(),
                    self.conserved0.as_ptr(),
                    self.primitive1.as_ptr(),
                    self.primitive2.as_mut_ptr(),
                    a,
                    dt,
                    self.mode,
                )
            };
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, _time: f64, _setup: &dyn Setup) -> f64 {
            let mut wavespeeds = vec![0.0; self.num_zones()];
            unsafe {
                euler1d_wavespeed(
                    self.num_zones() as i32,
                    self.primitive1.as_ptr(),
                    wavespeeds.as_mut_ptr(),
                    self.mode,
                )
            };
            wavespeeds.iter().cloned().fold(0.0, f64::max)
        }
    }
}
