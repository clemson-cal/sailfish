macro_rules! c_api {
    ($real:ty,
     $new:tt,
     $del:tt,
     $get_primitive:tt,
     $set_primitive:tt,
     $advance_cons:tt) => {
        use std::os::raw::c_void;

        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct CSolver(*mut c_void);

        pub struct Solver {
            raw: CSolver,
            mesh: Mesh,
        }

        impl Solver {
            pub fn new(mesh: Mesh) -> Self {
                Self {
                    raw: unsafe { solver_new(mesh.clone()) },
                    mesh,
                }
            }
            pub fn set_primitive(&mut self, primitive: &Vec<$real>) {
                let count = 3 * self.mesh.ni() * self.mesh.nj();
                assert! {
                    primitive.len() == count,
                    "primitive buffer has wrong size {}, expected {}",
                    primitive.len(),
                    count
                };
                unsafe { solver_set_primitive(self.raw, primitive.as_ptr()) }
            }
            pub fn primitive(&mut self) -> Vec<$real> {
                let count = 3 * self.mesh.ni() * self.mesh.nj();
                let mut primitive = vec![0.0; count];
                unsafe { solver_get_primitive(self.raw, primitive.as_mut_ptr()) }
                primitive
            }
            pub fn advance_cons(
                &mut self,
                eos: EquationOfState,
                buffer: BufferZone,
                masses: &Vec<PointMass>,
                dt: $real,
            ) {
                unsafe {
                    solver_advance_cons(self.raw, eos, buffer, masses.as_ptr(), masses.len() as u64, dt)
                }
            }
        }

        impl Drop for Solver {
            fn drop(&mut self) {
                unsafe { solver_del(self.raw) }
            }
        }

        extern "C" {
            #[link_name = $new]
            pub(crate) fn solver_new(mesh: Mesh) -> CSolver;
            #[link_name = $del]
            pub(crate) fn solver_del(solver: CSolver);
            #[link_name = $get_primitive]
            pub(crate) fn solver_get_primitive(solver: CSolver, primitive: *mut $real);
            #[link_name = $set_primitive]
            pub(crate) fn solver_set_primitive(solver: CSolver, primitive: *const $real);
            #[link_name = $advance_cons]
            pub(crate) fn solver_advance_cons(
                solver: CSolver,
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
            pub ni: u64,
            pub nj: u64,
            pub x0: $real,
            pub x1: $real,
            pub y0: $real,
            pub y1: $real,
        }

        impl Mesh {
            pub fn ni(&self) -> usize {
                self.ni as usize
            }
            pub fn nj(&self) -> usize {
                self.nj as usize
            }
            pub fn num_total_zones(&self) -> usize {
                (self.ni * self.nj) as usize
            }
            pub fn dx(&self) -> $real {
                (self.x1 - self.x0) / self.ni as $real
            }
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
                onset_radius: f64,
                onset_width: f64,
            }
        }
    }
}

pub mod f32 {
    mesh_struct!(f32);
    equation_of_state_struct!(f32);
    point_mass_struct!(f32);
    buffer_zone_struct!(f32);
    pub mod iso2d_cpu {
        use super::*;
        c_api! {
            f32,
            "iso2d_cpu_f32_solver_new",
            "iso2d_cpu_f32_solver_del",
            "iso2d_cpu_f32_solver_get_primitive",
            "iso2d_cpu_f32_solver_set_primitive",
            "iso2d_cpu_f32_solver_advance_cons"
        }
    }
}

pub mod f64 {
    mesh_struct!(f64);
    equation_of_state_struct!(f64);
    point_mass_struct!(f64);
    buffer_zone_struct!(f64);
    pub mod iso2d_cpu {
        use super::*;
        c_api! {
            f64,
            "iso2d_cpu_f64_solver_new",
            "iso2d_cpu_f64_solver_del",
            "iso2d_cpu_f64_solver_get_primitive",
            "iso2d_cpu_f64_solver_set_primitive",
            "iso2d_cpu_f64_solver_advance_cons"
        }
    }
}
