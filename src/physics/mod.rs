macro_rules! c_api {
    ($real:ty,
     $new:tt,
     $del:tt,
     $get_primitive:tt,
     $set_primitive:tt,
     $get_mesh:tt,
     $compute_fluxes:tt,
     $advance:tt) => {
        use std::os::raw::c_void;
        use super::*;

        #[repr(C)]
        pub struct Solver(*mut c_void);

        impl Solver {
            /// Creates a new solver instance from a mesh instance.
            pub fn new(mesh: Mesh) -> Self {
                unsafe { Self(solver_new(mesh.clone())) }
            }
        }

        impl Solve for Solver {
            /// Sets the primitive variable array in the solver. The number of
            /// elements in the input buffer must match the number of zones,
            /// times the number of primitive variables per zone.
            fn set_primitive(&mut self, primitive: &[$real]) {
                let count = 3 * self.mesh().num_total_zones();
                assert! {
                    primitive.len() == count,
                    "primitive buffer has wrong size {}, expected {}",
                    primitive.len(),
                    count
                };
                unsafe { solver_set_primitive(self.0, primitive.as_ptr()) }
            }

            /// Makes a deep copy of the primitive variable array in the
            /// solver and returns it as a vector.
            fn primitive(&mut self) -> Vec<$real> {
                let count = 3 * self.mesh().num_total_zones();
                let mut primitive = vec![0.0; count];
                unsafe { solver_get_primitive(self.0, primitive.as_mut_ptr()) }
                primitive
            }

            /// Retrieve a copy of the mesh struct used to create this solver
            /// instance.
            fn mesh(&self) -> Mesh {
                unsafe { solver_get_mesh(self.0) }
            }

            /// Pre-computes the Godunov fluxes on zone interfaces. If this
            /// function is called before `Solver::advance`, then the
            /// pre-computed Godunov fluxes will be loaded from buffers rather
            /// than computed in-place. The performance trade-off has to be
            /// checked empirically: especially on the GPU, it can be faster
            /// to compute fluxes with two-fold redundancy than to load them
            /// from global memory.
            fn compute_fluxes(
                &mut self,
                eos: EquationOfState,
                masses: &[PointMass],
            ) {
                unsafe {
                    solver_compute_fluxes(self.0, eos, masses.as_ptr(), masses.len() as u64)
                }
            }

            /// Advances the internal state of the solver by the given time
            /// step. Fluxes will be computed on-the-fly if they were not
            /// pre-computed.
            fn advance(
                &mut self,
                eos: EquationOfState,
                buffer: BufferZone,
                masses: &[PointMass],
                dt: $real,
            ) {
                unsafe {
                    solver_advance(self.0, eos, buffer, masses.as_ptr(), masses.len() as u64, dt)
                }
            }
        }

        impl Drop for Solver {
            fn drop(&mut self) {
                unsafe { solver_del(self.0) }
            }
        }

        extern "C" {
            #[link_name = $new]
            pub(crate) fn solver_new(mesh: Mesh) -> *mut c_void;
            #[link_name = $del]
            pub(crate) fn solver_del(solver: *mut c_void);
            #[link_name = $get_primitive]
            pub(crate) fn solver_get_primitive(solver: *mut c_void, primitive: *mut $real);
            #[link_name = $set_primitive]
            pub(crate) fn solver_set_primitive(solver: *mut c_void, primitive: *const $real);
            #[link_name = $get_mesh]
            pub(crate) fn solver_get_mesh(solver: *mut c_void) -> Mesh;
            #[link_name = $compute_fluxes]
            pub(crate) fn solver_compute_fluxes(
                solver: *mut c_void,
                eos: EquationOfState,
                masses: *const PointMass,
                num_masses: u64,
            );
            #[link_name = $advance]
            pub(crate) fn solver_advance(
                solver: *mut c_void,
                eos: EquationOfState,
                buffer: BufferZone,
                masses: *const PointMass,
                num_masses: u64,
                dt: $real,
            );
        }
    };
}

macro_rules! mesh_struct {
    ($real:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone)]
        pub struct Mesh {
            /// Number of zones on the i-axis
            pub ni: u64,
            /// Number of zones on the j-axis
            pub nj: u64,
            /// Left coordinate edge of the domain
            pub x0: $real,
            /// Right coordinate edge of the domain
            pub x1: $real,
            /// Bottom coordinate edge of the domain
            pub y0: $real,
            /// Top coordinate edge of the domain
            pub y1: $real,
        }

        impl Mesh {
            /// Returns the number of zones on the i-axis as a `usize`.
            pub fn ni(&self) -> usize {
                self.ni as usize
            }
            /// Returns the number of zones on the j-axis as a `usize`.
            pub fn nj(&self) -> usize {
                self.nj as usize
            }
            /// Returns the number of total zones (`ni * nj`) as a `usize`.
            pub fn num_total_zones(&self) -> usize {
                (self.ni * self.nj) as usize
            }
            /// Returns the grid spacing on the i-axis.
            pub fn dx(&self) -> $real {
                (self.x1 - self.x0) / self.ni as $real
            }
            /// Returns the grid spacing on the j-axis.
            pub fn dy(&self) -> $real {
                (self.y1 - self.y0) / self.nj as $real
            }
        }
    };
}

macro_rules! equation_of_state_struct {
    ($real:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub enum EquationOfState {
            Isothermal { sound_speed: $real },
            LocallyIsothermal { mach_number: $real },
            GammaLaw { gamma_law_index: $real },
        }
    };
}

macro_rules! point_mass_struct {
    ($real:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct PointMass {
            pub x: $real,
            pub y: $real,
            pub mass: $real,
            pub rate: $real,
            pub radius: $real,
        }
    };
}

macro_rules! buffer_zone_struct {
    ($real:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub enum BufferZone {
            None,
            Keplerian {
                surface_density: f64,
                central_mass: f64,
                driving_rate: f64,
                outer_radius: f64,
                onset_width: f64,
            }
        }
    }
}

macro_rules! solve_trait {
    ($real:ty) => {
        pub trait Solve {
            fn set_primitive(&mut self, primitive: &[$real]);
            fn primitive(&mut self) -> Vec<$real>;
            fn mesh(&self) -> Mesh;
            fn compute_fluxes(
                &mut self,
                eos: EquationOfState,
                masses: &[PointMass],
            );
            fn advance(
                &mut self,
                eos: EquationOfState,
                buffer: BufferZone,
                masses: &[PointMass],
                dt: $real,
            );
        }
    }
}

pub mod f32 {
    mesh_struct!(f32);
    equation_of_state_struct!(f32);
    point_mass_struct!(f32);
    buffer_zone_struct!(f32);
    solve_trait!(f32);

    pub mod iso2d_cpu {
        c_api! {
            f32,
            "iso2d_cpu_f32_solver_new",
            "iso2d_cpu_f32_solver_del",
            "iso2d_cpu_f32_solver_get_primitive",
            "iso2d_cpu_f32_solver_set_primitive",
            "iso2d_cpu_f32_solver_get_mesh",
            "iso2d_cpu_f32_solver_compute_fluxes",
            "iso2d_cpu_f32_solver_advance"
        }
    }

    #[cfg(feature = "omp")]
    pub mod iso2d_omp {
        c_api! {
            f32,
            "iso2d_omp_f32_solver_new",
            "iso2d_omp_f32_solver_del",
            "iso2d_omp_f32_solver_get_primitive",
            "iso2d_omp_f32_solver_set_primitive",
            "iso2d_omp_f32_solver_get_mesh",
            "iso2d_omp_f32_solver_compute_fluxes",
            "iso2d_omp_f32_solver_advance"
        }
    }

    #[cfg(feature = "cuda")]
    pub mod iso2d_cuda {
        c_api! {
            f32,
            "iso2d_cuda_f32_solver_new",
            "iso2d_cuda_f32_solver_del",
            "iso2d_cuda_f32_solver_get_primitive",
            "iso2d_cuda_f32_solver_set_primitive",
            "iso2d_cuda_f32_solver_get_mesh",
            "iso2d_cuda_f32_solver_compute_fluxes",
            "iso2d_cuda_f32_solver_advance"
        }
    }
}

pub mod f64 {
    mesh_struct!(f64);
    equation_of_state_struct!(f64);
    point_mass_struct!(f64);
    buffer_zone_struct!(f64);
    solve_trait!(f64);

    pub mod iso2d_cpu {
        c_api! {
            f64,
            "iso2d_cpu_f64_solver_new",
            "iso2d_cpu_f64_solver_del",
            "iso2d_cpu_f64_solver_get_primitive",
            "iso2d_cpu_f64_solver_set_primitive",
            "iso2d_cpu_f64_solver_get_mesh",
            "iso2d_cpu_f64_solver_compute_fluxes",
            "iso2d_cpu_f64_solver_advance"
        }
    }

    #[cfg(feature = "omp")]
    pub mod iso2d_omp {
        c_api! {
            f64,
            "iso2d_omp_f64_solver_new",
            "iso2d_omp_f64_solver_del",
            "iso2d_omp_f64_solver_get_primitive",
            "iso2d_omp_f64_solver_set_primitive",
            "iso2d_omp_f64_solver_get_mesh",
            "iso2d_omp_f64_solver_compute_fluxes",
            "iso2d_omp_f64_solver_advance"
        }
    }

    #[cfg(feature = "cuda")]
    pub mod iso2d_cuda {
        c_api! {
            f64,
            "iso2d_cuda_f64_solver_new",
            "iso2d_cuda_f64_solver_del",
            "iso2d_cuda_f64_solver_get_primitive",
            "iso2d_cuda_f64_solver_set_primitive",
            "iso2d_cuda_f64_solver_get_mesh",
            "iso2d_cuda_f64_solver_compute_fluxes",
            "iso2d_cuda_f64_solver_advance"
        }
    }
}
