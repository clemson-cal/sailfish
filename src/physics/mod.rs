#[repr(C)]
#[derive(Clone)]
pub struct Mesh {
    pub ni: u64,
    pub nj: u64,
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
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
    pub fn dx(&self) -> f64 {
        (self.x1 - self.x0) / self.ni as f64
    }
    pub fn dy(&self) -> f64 {
        (self.y1 - self.y0) / self.nj as f64
    }
}

macro_rules! solver_module {
    ($mod:ident,
     $real:ty,
     $new:tt,
     $del:tt,
     $get_primitive:tt,
     $set_primitive:tt,
     $advance_cons:tt) => {
        pub mod $mod {
            use super::Mesh;
            use std::os::raw::c_void;

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
                pub(crate) fn solver_advance_cons(solver: CSolver, dt: $real);
            }

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
    "iso2d_cpu_f32_solver_get_primitive",
    "iso2d_cpu_f32_solver_set_primitive",
    "iso2d_cpu_f32_solver_advance_cons"
);

solver_module!(
    iso2d_cpu_f64,
    f64,
    "iso2d_cpu_f64_solver_new",
    "iso2d_cpu_f64_solver_del",
    "iso2d_cpu_f64_solver_get_primitive",
    "iso2d_cpu_f64_solver_set_primitive",
    "iso2d_cpu_f64_solver_advance_cons"
);
