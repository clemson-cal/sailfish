#![allow(unused)]

#[repr(C)]
#[derive(Clone)]
pub struct Configuration {
    pub grid_dim: u64,
    pub sink_rate: f64,
    pub sink_radius: f64,
    pub mach_number: f64,
    pub domain_radius: f64,
}

impl Configuration {
    pub fn ni(&self) -> usize {
        self.grid_dim as usize
    }
    pub fn nj(&self) -> usize {
        self.grid_dim as usize
    }
}

macro_rules! solver_module {
    ($mod:ident, $real:ty, $new:tt, $del:tt, $set_primitive:tt, $advance_cons:tt) => {
        pub mod $mod {
            use super::Configuration;
            use std::os::raw::c_void;

            extern "C" {
                #[link_name = $new]
                pub(crate) fn solver_new(config: Configuration) -> CSolver;
                #[link_name = $del]
                pub(crate) fn solver_del(solver: CSolver);
                #[link_name = $set_primitive]
                pub(crate) fn solver_set_primitive(solver: CSolver, primitive: *const $real);
                #[link_name = $advance_cons]
                pub(crate) fn solver_advance_cons(solver: CSolver, dt: $real);
            }

            #[repr(C)]
            #[derive(Copy, Clone)]
            pub struct CSolver(*mut c_void);

            pub struct Solver {
                raw: CSolver,
                config: Configuration,
            }

            impl Solver {
                pub fn new(config: Configuration) -> Self {
                    Self {
                        raw: unsafe { solver_new(config.clone()) },
                        config: config,
                    }
                }
                pub fn set_primitive(&mut self, primitive: &Vec<$real>) {
                    let num_expected = 3 * self.config.grid_dim.pow(2) as usize;

                    assert!{
                        primitive.len() == num_expected,
                        "primitive buffer has wrong size {}, expected {}",
                        primitive.len(),
                        num_expected
                    };
                    unsafe { solver_set_primitive(self.raw, primitive.as_ptr()) }
                }
                pub fn advance_cons(&mut self, dt: $real) {
                    unsafe { solver_advance_cons(self.raw, dt) }
                }
            }

            impl Drop for Solver {
                fn drop(&mut self) {
                    unsafe { solver_del(self.raw) }
                }
            }
        }
    };
}

solver_module!(
    iso2d_cpu_f32,
    f32,
    "iso2d_cpu_f32_solver_new",
    "iso2d_cpu_f32_solver_del",
    "iso2d_cpu_f32_solver_set_primitive",
    "iso2d_cpu_f32_solver_advance_cons"
);

solver_module!(
    iso2d_cpu_f64,
    f64,
    "iso2d_cpu_f64_solver_new",
    "iso2d_cpu_f64_solver_del",
    "iso2d_cpu_f64_solver_set_primitive",
    "iso2d_cpu_f64_solver_advance_cons"
);
