use crate::sailfish::Mesh;
use crate::sailfish::ExecutionMode;

extern "C" {
    pub fn iso2d_primitive_to_conserved(mesh: Mesh, primitive_ptr: *const f64, conserved_ptr: *mut f64, mode: ExecutionMode);
}
