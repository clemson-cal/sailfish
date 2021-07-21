use crate::sailfish::{
    BufferZone, EquationOfState, ExecutionMode, PointMass, Solve, StructuredMesh,
};
use crate::Setup;
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

pub fn solver(
    mode: ExecutionMode,
    device: Option<i32>,
    mesh: StructuredMesh,
    primitive: &[f64],
) -> Box<dyn Solve> {
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
                if #[cfg(feature = "gpu")] {
                    Box::new(gpu::Solver::new(device, mesh, primitive))
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

    #[derive(Debug)]
    pub struct Solver {
        mesh: StructuredMesh,
        primitive1: Vec<f64>,
        primitive2: Vec<f64>,
        conserved0: Vec<f64>,
        pub(super) mode: ExecutionMode,
    }

    impl Solver {
        pub fn new(mesh: StructuredMesh, primitive: &[f64]) -> Self {
            assert_eq!(
                primitive.len(),
                (mesh.ni as usize + 4) * (mesh.nj as usize + 4) * 3
            );
            Self {
                mesh,
                primitive1: primitive.to_vec(),
                primitive2: primitive.to_vec(),
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
            setup: &dyn Setup,
            time: f64,
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
        fn max_wavespeed(&self, time: f64, setup: &dyn Setup) -> f64 {
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
        pub fn new(mesh: StructuredMesh, primitive: &[f64]) -> Self {
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
    use gpu_core::{Device, DeviceBuffer};

    pub struct Solver {
        mesh: StructuredMesh,
        primitive1: DeviceBuffer<f64>,
        primitive2: DeviceBuffer<f64>,
        conserved0: DeviceBuffer<f64>,
        wavespeeds: DeviceBuffer<f64>,
        device: Device,
    }

    impl Solver {
        pub fn new(device: Option<i32>, mesh: StructuredMesh, primitive: &[f64]) -> Self {
            assert_eq!(
                primitive.len(),
                (mesh.ni as usize + 4) * (mesh.nj as usize + 4) * 3
            );
            let device = Device::with_id(device.unwrap_or(0)).expect("invalid device id");
            Self {
                mesh,
                primitive1: device.buffer_from(primitive),
                primitive2: device.buffer_from(primitive),
                conserved0: device.buffer_from(&vec![0.0; mesh.num_total_zones() * 3]),
                wavespeeds: device.buffer_from(&vec![0.0; mesh.num_total_zones()]),
                device,
            }
        }
    }

    impl Solve for Solver {
        fn primitive(&self) -> Vec<f64> {
            Vec::from(&self.primitive1)
        }
        fn primitive_to_conserved(&mut self) {
            self.device.scope(|_| {
                unsafe {
                    iso2d_primitive_to_conserved(
                        self.mesh,
                        self.primitive1.as_device_ptr(),
                        self.conserved0.as_device_ptr() as *mut f64,
                        ExecutionMode::GPU,
                    );
                }
            })
        }
        fn advance_rk(
            &mut self,
            setup: &dyn Setup,
            time: f64,
            a: f64,
            dt: f64,
            velocity_ceiling: f64,
        ) {
            self.device.scope(|device| {
                let buffer = setup.buffer_zone();
                let eos = setup.equation_of_state();
                let nu = setup.viscosity().unwrap_or(0.0);
                let masses = device.buffer_from(&setup.masses(time));
                unsafe {
                    iso2d_advance_rk(
                        self.mesh,
                        self.conserved0.as_device_ptr(),
                        self.primitive1.as_device_ptr(),
                        self.primitive2.as_device_ptr() as *mut f64,
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
            });
            std::mem::swap(&mut self.primitive1, &mut self.primitive2);
        }
        fn max_wavespeed(&self, time: f64, setup: &dyn Setup) -> f64 {
            use gpu_core::Reduce;
            self.device.scope(|device| {
                let eos = setup.equation_of_state();
                let masses = device.buffer_from(&setup.masses(time));

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
            })
        }
    }
}
