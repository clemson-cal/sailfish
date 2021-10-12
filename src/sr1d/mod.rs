use crate::error::Error;
use crate::{BoundaryCondition, Coordinates, ExecutionMode, Setup, Solve};
use cfg_if::cfg_if;

extern "C" {
    fn sr1d_primitive_to_conserved(
        num_zones: i32,
        face_positions_ptr: *const f64,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        scale_factor: f64,
        coords: Coordinates,
        mode: ExecutionMode,
    );

    fn sr1d_conserved_to_primitive(
        num_zones: i32,
        face_positions_ptr: *const f64,
        conserved_ptr: *const f64,
        primitive_ptr: *mut f64,
        scale_factor: f64,
        coords: Coordinates,
        mode: ExecutionMode,
    );

    fn sr1d_advance_rk(
        num_zones: i32,
        face_positions_ptr: *const f64,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        conserved_rd_ptr: *const f64,
        conserved_wr_ptr: *mut f64,
        a0: f64,
        adot: f64,
        time: f64,
        a: f64,
        dt: f64,
        boundary_condition: BoundaryCondition,
        coords: Coordinates,
        mode: ExecutionMode,
    );
}

pub fn solver(
    mode: ExecutionMode,
    device: Option<i32>,
    faces: &[f64],
    primitive: &[f64],
    homologous_expansion: Option<(f64, f64)>,
    boundary_condition: BoundaryCondition,
    coords: Coordinates,
    scale_factor: f64,
) -> Result<Box<dyn Solve>, Error> {
    let homologous_expansion = homologous_expansion.unwrap_or((1.0, 0.0));
    match mode {
        ExecutionMode::CPU => Ok(Box::new(cpu::Solver::new(
            faces,
            primitive,
            homologous_expansion,
            boundary_condition,
            coords,
            scale_factor,
        ))),
        ExecutionMode::OMP => {
            cfg_if! {
                if #[cfg(feature = "omp")] {
                    Ok(Box::new(omp::Solver::new(
                        faces,
                        primitive,
                        homologous_expansion,
                        boundary_condition,
                        coords,
                        scale_factor)))
                } else {
                    panic!()
                }
            }
        }
        ExecutionMode::GPU => {
            cfg_if! {
                if #[cfg(feature = "gpu")] {
                    Ok(Box::new(gpu::Solver::new(
                        device,
                        faces,
                        primitive,
                        homologous_expansion,
                        boundary_condition,
                        coords,
                        scale_factor)?))
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
        conserved1: Vec<f64>,
        conserved2: Vec<f64>,
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
            scale_factor: f64,
        ) -> Self {
            let num_zones = faces.len() - 1;
            let mut conserved = vec![0.0; primitive.len()];
            unsafe {
                sr1d_primitive_to_conserved(
                    num_zones as i32,
                    faces.as_ptr(),
                    primitive.as_ptr(),
                    conserved.as_mut_ptr(),
                    scale_factor,
                    coords,
                    ExecutionMode::CPU,
                );
            }
            assert_eq!(primitive.len(), num_zones * 3);
            Self {
                faces: faces.to_vec(),
                primitive1: primitive.to_vec(),
                conserved1: conserved.clone(),
                conserved2: conserved,
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

        fn scale_factor(&self, time: f64) -> f64 {
            let (a0, a1) = self.homologous_parameters;
            a0 + a1 * time
        }

        fn recompute_primitive(&self, time: f64) {
            unsafe {
                sr1d_conserved_to_primitive(
                    self.num_zones() as i32,
                    self.faces.as_ptr(),
                    self.conserved1.as_ptr(),
                    self.primitive1.as_ptr() as *mut f64,
                    self.scale_factor(time),
                    self.coords,
                    self.mode,
                );
            }
        }
    }

    impl Solve for Solver {
        fn primitive(&self, time: f64) -> Vec<f64> {
            self.recompute_primitive(time);
            self.primitive1.to_vec()
        }
        fn new_timestep(&mut self, _time: f64) {
            self.conserved0.copy_from_slice(&self.conserved1);
        }
        fn advance_rk(&mut self, _setup: &dyn Setup, time: f64, a: f64, dt: f64) {
            self.recompute_primitive(time);
            unsafe {
                sr1d_advance_rk(
                    self.num_zones() as i32,
                    self.faces.as_ptr(),
                    self.conserved0.as_ptr(),
                    self.primitive1.as_ptr(),
                    self.conserved1.as_ptr(),
                    self.conserved2.as_mut_ptr(),
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
            std::mem::swap(&mut self.conserved1, &mut self.conserved2);
        }
        fn max_wavespeed(&self, _time: f64, _setup: &dyn Setup) -> f64 {
            1.0
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
            scale_factor: f64,
        ) -> Self {
            let mut solver = cpu::Solver::new(
                faces,
                primitive,
                homologous_parameters,
                boundary_condition,
                coords,
                scale_factor,
            );
            solver.mode = ExecutionMode::OMP;
            Self(solver)
        }
    }

    impl Solve for Solver {
        fn primitive(&self, time: f64) -> Vec<f64> {
            self.0.primitive(time)
        }
        fn new_timestep(&mut self, time: f64) {
            self.0.new_timestep(time)
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
        conserved0: DeviceBuffer<f64>,
        primitive1: DeviceBuffer<f64>,
        conserved1: DeviceBuffer<f64>,
        conserved2: DeviceBuffer<f64>,
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
            scale_factor: f64,
        ) -> Result<Self, Error> {
            let num_zones = faces.len() - 1;
            assert_eq!(primitive.len(), num_zones * 3);

            let mut conserved = vec![0.0; primitive.len()];
            unsafe {
                sr1d_primitive_to_conserved(
                    num_zones as i32,
                    faces.as_ptr(),
                    primitive.as_ptr(),
                    conserved.as_mut_ptr(),
                    scale_factor,
                    coords,
                    ExecutionMode::CPU,
                );
            }

            let id = device.unwrap_or(0);
            let device = Device::with_id(id).ok_or(Error::InvalidDevice(id))?;
            Ok(Self {
                faces: device.buffer_from(faces),
                conserved0: device.buffer_from(&conserved),
                primitive1: device.buffer_from(&primitive),
                conserved1: device.buffer_from(&conserved),
                conserved2: device.buffer_from(&conserved),
                homologous_parameters,
                boundary_condition,
                coords,
                device,
            })
        }

        fn num_zones(&self) -> usize {
            self.faces.len() - 1
        }

        fn scale_factor(&self, time: f64) -> f64 {
            let (a0, a1) = self.homologous_parameters;
            a0 + a1 * time
        }

        fn recompute_primitive(&self, time: f64) {
            self.device.scope(|_| {
                unsafe {
                    sr1d_conserved_to_primitive(
                        self.num_zones() as i32,
                        self.faces.as_device_ptr(),
                        self.conserved1.as_device_ptr(),
                        self.primitive1.as_device_ptr() as *mut f64,
                        self.scale_factor(time),
                        self.coords,
                        ExecutionMode::GPU,
                    );
                }
            })
        }
    }

    impl Solve for Solver {
        fn primitive(&self, time: f64) -> Vec<f64> {
            self.recompute_primitive(time);
            self.primitive1.to_vec()
        }
        fn new_timestep(&mut self, _time: f64) {
            self.conserved1.copy_into(&mut self.conserved0);
        }
        fn advance_rk(&mut self, _setup: &dyn Setup, time: f64, a: f64, dt: f64) {
            self.recompute_primitive(time);
            self.device.scope(|_| {
                unsafe {
                    sr1d_advance_rk(
                        self.num_zones() as i32,
                        self.faces.as_device_ptr(),
                        self.conserved0.as_device_ptr(),
                        self.primitive1.as_device_ptr(),
                        self.conserved1.as_device_ptr(),
                        self.conserved2.as_device_ptr() as *mut f64,
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
            std::mem::swap(&mut self.conserved1, &mut self.conserved2);
        }
        fn max_wavespeed(&self, _time: f64, _setup: &dyn Setup) -> f64 {
            1.0
        }
    }
}
