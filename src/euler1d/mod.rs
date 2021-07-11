use crate::Setup;
use crate::sailfish::{ExecutionMode, Solve};
use cfg_if::cfg_if;

extern "C" {
    fn euler1d_primitive_to_conserved(
        num_zones: i32,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn euler1d_advance_rk(
        num_zones: i32,
        face_positions_ptr: *const f64,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        a: f64,
        dt: f64,
        mode: ExecutionMode,
    );

    #[cfg(feature = "gpu")]
    fn euler1d_wavespeed(
        num_zones: i32,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn euler1d_max_wavespeed(num_zones: i32, primitive_ptr: *const f64, mode: ExecutionMode)
        -> f64;
}

pub fn solver(mode: ExecutionMode, faces: &[f64], primitive: &[f64]) -> Box<dyn Solve> {
    match mode {
        ExecutionMode::CPU => Box::new(cpu::Solver::new(faces, primitive)),
        ExecutionMode::OMP => {
            cfg_if! {
                if #[cfg(feature = "omp")] {
                    Box::new(omp::Solver::new(faces, primitive))
                } else {
                    panic!()
                }
            }
        }
        ExecutionMode::GPU => {
            cfg_if! {
                if #[cfg(feature = "gpu")] {
                    Box::new(gpu::Solver::new(faces, primitive))
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
        faces: Vec<f64>,
        primitive1: Vec<f64>,
        primitive2: Vec<f64>,
        conserved0: Vec<f64>,
        pub(super) mode: ExecutionMode,
    }

    impl Solver {
        pub fn new(faces: &[f64], primitive: &[f64]) -> Self {
            let num_zones = faces.len() - 1;
            assert_eq!(primitive.len(), num_zones * 3);
            Self {
                faces: faces.to_vec(),
                primitive1: primitive.to_vec(),
                primitive2: primitive.to_vec(),
                conserved0: vec![0.0; num_zones * 3],
                mode: ExecutionMode::CPU,
            }
        }

        fn num_zones(&self) -> usize {
            self.faces.len() - 1
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
                    self.faces.as_ptr(),
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
            unsafe {
                euler1d_max_wavespeed(self.num_zones() as i32, self.primitive1.as_ptr(), self.mode)
            }
        }
    }
}

pub mod omp {
    use super::*;
    pub struct Solver(cpu::Solver);

    impl Solver {
        pub fn new(faces: &[f64], primitive: &[f64]) -> Self {
            let mut solver = cpu::Solver::new(faces, primitive);
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
            setup: &dyn Setup,
            time: f64,
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            self.0.advance_rk(setup, time, a, dt, velocity_ceiling)
        }
        fn max_wavespeed(&self, time: f64, setup: &dyn Setup) -> f64 {
            self.0.max_wavespeed(time, setup)
        }
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use gpu_mem::DeviceVec;

    pub struct Solver {
        faces: DeviceVec<f64>,
        primitive1: DeviceVec<f64>,
        primitive2: DeviceVec<f64>,
        conserved0: DeviceVec<f64>,
        wavespeeds: DeviceVec<f64>,
    }

    impl Solver {
        pub fn new(faces: &[f64], primitive: &[f64]) -> Self {
            let num_zones = faces.len() - 1;
            assert_eq!(primitive.len(), num_zones * 3);
            Self {
                faces: DeviceVec::from(faces),
                primitive1: DeviceVec::from(primitive),
                primitive2: DeviceVec::from(primitive),
                conserved0: DeviceVec::from(&vec![0.0; num_zones * 3]),
                wavespeeds: DeviceVec::from(&vec![0.0; num_zones]),
            }
        }

        fn num_zones(&self) -> usize {
            self.faces.len() - 1
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            Vec::from(&self.primitive1)
        }
        fn primitive_to_conserved(&mut self) {
            unsafe {
                euler1d_primitive_to_conserved(
                    self.num_zones() as i32,
                    self.primitive1.as_device_ptr(),
                    self.conserved0.as_mut_device_ptr(),
                    ExecutionMode::GPU,
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
                    self.faces.as_device_ptr(),
                    self.conserved0.as_device_ptr(),
                    self.primitive1.as_device_ptr(),
                    self.primitive2.as_mut_device_ptr(),
                    a,
                    dt,
                    ExecutionMode::GPU,
                )
            };
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, _time: f64, _setup: &dyn Setup) -> f64 {
            use gpu_mem::Reduce;
            unsafe {
                euler1d_wavespeed(
                    self.num_zones() as i32,
                    self.primitive1.as_device_ptr(),
                    self.wavespeeds.as_device_ptr() as *mut f64,
                    ExecutionMode::GPU,
                )
            };
            self.wavespeeds.maximum().unwrap()
        }
    }
}
