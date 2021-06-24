use super::patch::{device, ffi};

extern "C" {
    fn primitive_to_conserved_cpu(primitive: ffi::Patch, conserved: ffi::Patch);
    fn conserved_to_primitive_cpu(conserved: ffi::Patch, primitive: ffi::Patch);
}

pub fn primitive_to_conserved(primitive: &device::Patch, conserved: &mut device::Patch) {
    unsafe {
        primitive_to_conserved_cpu(primitive.0, conserved.0)
    }
}

pub fn conserved_to_primitive(conserved: &device::Patch, primitive: &mut device::Patch) {
    unsafe {
        conserved_to_primitive_cpu(conserved.0, primitive.0)
    }
}
