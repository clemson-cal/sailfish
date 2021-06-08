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

pub mod iso2d_cpu_f32 {
    use super::Configuration;
    use crate::physics::iso2d_cpu_f32;
    use std::os::raw::c_void;

    extern "C" {
        pub(crate) fn iso2d_cpu_f32_solver_new(config: Configuration) -> CSolver;
        pub(crate) fn iso2d_cpu_f32_solver_del(solver: CSolver);
        pub(crate) fn iso2d_cpu_f32_solver_set_primitive(solver: CSolver, primitive: *const f32);
        pub(crate) fn iso2d_cpu_f32_solver_do_advance_cons(solver: CSolver, dt: f32);
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
                raw: unsafe { iso2d_cpu_f32_solver_new(config.clone()) },
                config: config,
            }
        }
        pub fn set_primitive(&mut self, primitive: &Vec<f32>) {
            let num_expected = 3 * self.config.grid_dim.pow(2) as usize;

            assert!(
                primitive.len() == num_expected,
                "primitive buffer has wrong size {}, expected {}",
                primitive.len(),
                num_expected
            );
            unsafe { iso2d_cpu_f32_solver_set_primitive(self.raw, primitive.as_ptr()) }
        }
        pub fn advance_cons(&mut self, dt: f32) {
            unsafe { iso2d_cpu_f32_solver_do_advance_cons(self.raw, dt) }
        }
    }

    impl Drop for Solver {
        fn drop(&mut self) {
            unsafe { iso2d_cpu_f32_solver_del(self.raw) }
        }
    }
}
