use crate::error::Error;
use crate::{BoundaryCondition, Coordinates, ExecutionMode, Setup, Solve};
use cfg_if::cfg_if;

extern "C" {
    fn sr1d_primitive_to_conserved(
        num_zones: i32,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn sr1d_advance_rk(
        num_zones: i32,
        face_positions_ptr: *const f64,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        a0: f64,
        adot: f64,
        t: f64,
        a: f64,
        dt: f64,
        boundary_condition: BoundaryCondition,
        coords: Coordinates,
        mode: ExecutionMode,
    );

    #[cfg(feature = "gpu")]
    fn sr1d_wavespeed(
        num_zones: i32,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn sr1d_max_wavespeed(num_zones: i32, primitive_ptr: *const f64, mode: ExecutionMode) -> f64;
}

pub fn solver(
    mode: ExecutionMode,
    device: Option<i32>,
    faces: &[f64],
    primitive: &[f64],
    homologous_expansion: Option<(f64, f64)>,
    boundary_condition: BoundaryCondition,
    coords: Coordinates,
) -> Result<Box<dyn Solve>, Error> {
    let homologous_expansion = homologous_expansion.unwrap_or((1.0, 0.0));
    match mode {
        ExecutionMode::CPU => Ok(Box::new(cpu::Solver::new(
            faces,
            primitive,
            homologous_expansion,
            boundary_condition,
            coords,
        ))),
        ExecutionMode::OMP => {
            cfg_if! {
                if #[cfg(feature = "omp")] {
                    Ok(Box::new(omp::Solver::new(faces, primitive, homologous_expansion, boundary_condition, coords)))
                } else {
                    panic!()
                }
            }
        }
        ExecutionMode::GPU => {
            cfg_if! {
                if #[cfg(feature = "gpu")] {
                    Ok(Box::new(gpu::Solver::new(device, faces, primitive, homologous_expansion, boundary_condition, coords)?))
                } else {
                    std::convert::identity(device); // black-box
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
        homologous_parameters: (f64, f64),
        boundary_condition: BoundaryCondition,
        coords: Coordinates,
        pub(super) mode: ExecutionMode,
    }

    impl Solver {
        pub fn new(
            faces: &[f64],
            primitive: &[f64],
            homologous_parameters: (f64, f64),
            boundary_condition: BoundaryCondition,
            coords: Coordinates,
        ) -> Self {
            let num_zones = faces.len() - 1;
            assert_eq!(primitive.len(), num_zones * 3);
            Self {
                faces: faces.to_vec(),
                primitive1: primitive.to_vec(),
                primitive2: primitive.to_vec(),
                conserved0: vec![0.0; num_zones * 3],
                homologous_parameters,
                boundary_condition,
                coords,
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
                sr1d_primitive_to_conserved(
                    self.num_zones() as i32,
                    self.primitive1.as_ptr(),
                    self.conserved0.as_mut_ptr(),
                    self.mode,
                );
            }
        }
        fn advance_rk(&mut self, _setup: &dyn Setup, time: f64, a: f64, dt: f64) {
            unsafe {
                sr1d_advance_rk(
                    self.num_zones() as i32,
                    self.faces.as_ptr(),
                    self.conserved0.as_ptr(),
                    self.primitive1.as_ptr(),
                    self.primitive2.as_mut_ptr(),
                    self.homologous_parameters.0,
                    self.homologous_parameters.1,
                    time,
                    a,
                    dt,
                    self.boundary_condition,
                    self.coords,
                    self.mode,
                )
            };
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, _time: f64, _setup: &dyn Setup) -> f64 {
            unsafe {
                sr1d_max_wavespeed(self.num_zones() as i32, self.primitive1.as_ptr(), self.mode)
            }
        }
    }
}

pub mod omp {
    use super::*;
    pub struct Solver(cpu::Solver);

    impl Solver {
        pub fn new(
            faces: &[f64],
            primitive: &[f64],
            homologous_parameters: (f64, f64),
            boundary_condition: BoundaryCondition,
            coords: Coordinates,
        ) -> Self {
            let mut solver = cpu::Solver::new(faces, primitive, homologous_parameters, boundary_condition, coords);
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
        fn advance_rk(&mut self, setup: &dyn Setup, time: f64, a: f64, dt: f64) {
            self.0.advance_rk(setup, time, a, dt)
        }
        fn max_wavespeed(&self, time: f64, setup: &dyn Setup) -> f64 {
            self.0.max_wavespeed(time, setup)
        }
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use gpu_core::{Device, DeviceBuffer};

    pub struct Solver {
        faces: DeviceBuffer<f64>,
        primitive1: DeviceBuffer<f64>,
        primitive2: DeviceBuffer<f64>,
        conserved0: DeviceBuffer<f64>,
        wavespeeds: DeviceBuffer<f64>,
        homologous_parameters: (f64, f64),
        boundary_condition: BoundaryCondition,
        coords: Coordinates,
        device: Device,
    }

    impl Solver {
        pub fn new(
            device: Option<i32>,
            faces: &[f64],
            primitive: &[f64],
            homologous_parameters: (f64, f64),
            boundary_condition: BoundaryCondition,
            coords: Coordinates,
        ) -> Result<Self, Error> {
            let num_zones = faces.len() - 1;
            assert_eq!(primitive.len(), num_zones * 3);
            let id = device.unwrap_or(0);
            let device = Device::with_id(id).ok_or(Error::InvalidDevice(id))?;
            Ok(Self {
                faces: device.buffer_from(faces),
                primitive1: device.buffer_from(primitive),
                primitive2: device.buffer_from(primitive),
                conserved0: device.buffer_from(&vec![0.0; num_zones * 3]),
                wavespeeds: device.buffer_from(&vec![0.0; num_zones]),
                homologous_parameters,
                boundary_condition,
                coords,
                device,
            })
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
            self.device.scope(|_| unsafe {
                sr1d_primitive_to_conserved(
                    self.num_zones() as i32,
                    self.primitive1.as_device_ptr(),
                    self.conserved0.as_device_ptr() as *mut f64,
                    ExecutionMode::GPU,
                );
            })
        }
        fn advance_rk(&mut self, _setup: &dyn Setup, time: f64, a: f64, dt: f64) {
            self.device.scope(|_| {
                unsafe {
                    sr1d_advance_rk(
                        self.num_zones() as i32,
                        self.faces.as_device_ptr(),
                        self.conserved0.as_device_ptr(),
                        self.primitive1.as_device_ptr(),
                        self.primitive2.as_device_ptr() as *mut f64,
                        self.homologous_parameters.0,
                        self.homologous_parameters.1,
                        time,
                        a,
                        dt,
                        self.boundary_condition,
                        self.coords,
                        ExecutionMode::GPU,
                    )
                };
            });
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, _time: f64, _setup: &dyn Setup) -> f64 {
            use gpu_core::Reduce;
            self.device.scope(|_| {
                unsafe {
                    sr1d_wavespeed(
                        self.num_zones() as i32,
                        self.primitive1.as_device_ptr(),
                        self.wavespeeds.as_device_ptr() as *mut f64,
                        ExecutionMode::GPU,
                    )
                };
            });
            self.wavespeeds.maximum().unwrap()
        }
    }
}
