use super::patch::{host, device, ffi};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mesh
{
    ni: i32,
    nj: i32,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
}

mod iso2d_ffi {
    use super::*;
    extern "C" {
        pub(super) fn primitive_to_conserved_cpu(primitive: ffi::Patch, conserved: ffi::Patch);
        pub(super) fn primitive_to_conserved_omp(primitive: ffi::Patch, conserved: ffi::Patch);
        pub(super) fn primitive_to_conserved_gpu(primitive: ffi::Patch, conserved: ffi::Patch);

        pub(super) fn advance_rk_cpu(
            mesh: Mesh,
            conserved_rk: ffi::Patch,
            primitive_rd: ffi::Patch,
            primitive_wr: ffi::Patch,
            a: f64,
            dt: f64);

        pub(super) fn advance_rk_omp(
            mesh: Mesh,
            conserved_rk: ffi::Patch,
            primitive_rd: ffi::Patch,
            primitive_wr: ffi::Patch,
            a: f64,
            dt: f64);

        pub(super) fn advance_rk_gpu(
            mesh: Mesh,
            conserved_rk: ffi::Patch,
            primitive_rd: ffi::Patch,
            primitive_wr: ffi::Patch,
            a: f64,
            dt: f64);
    }
}

pub fn primitive_to_conserved_cpu(primitive: &host::Patch, conserved: &mut host::Patch) {
    unsafe {
        iso2d_ffi::primitive_to_conserved_cpu(primitive.0, conserved.0)
    }
}

#[cfg(feature = "omp")]
pub fn primitive_to_conserved_omp(primitive: &host::Patch, conserved: &mut host::Patch) {
    unsafe {
        iso2d_ffi::primitive_to_conserved_omp(primitive.0, conserved.0)
    }
}

#[cfg(feature = "cuda")]
pub fn primitive_to_conserved_gpu(primitive: &device::Patch, conserved: &mut device::Patch) {
    unsafe {
        iso2d_ffi::primitive_to_conserved_gpu(primitive.0, conserved.0)
    }
}

pub fn advance_rk_cpu(
    mesh: Mesh,
    conserved_rk: &host::Patch,
    primitive_rd: &host::Patch,
    primitive_wr: &mut host::Patch,
    a: f64,
    dt: f64) {
    unsafe {
        iso2d_ffi::advance_rk_cpu(mesh, conserved_rk.0, primitive_rd.0, primitive_wr.0, a, dt)
    }
}

#[cfg(feature = "omp")]
pub fn advance_rk_omp(
    mesh: Mesh,
    conserved_rk: &host::Patch,
    primitive_rd: &host::Patch,
    primitive_wr: &mut host::Patch,
    a: f64,
    dt: f64) {
    unsafe {
        iso2d_ffi::advance_rk_omp(mesh, conserved_rk.0, primitive_rd.0, primitive_wr.0, a, dt)
    }
}

#[cfg(feature = "cuda")]
pub fn advance_rk_gpu(
    mesh: Mesh,
    conserved_rk: &device::Patch,
    primitive_rd: &device::Patch,
    primitive_wr: &mut device::Patch,
    a: f64,
    dt: f64) {
    unsafe {
        iso2d_ffi::advance_rk_gpu(mesh, conserved_rk.0, primitive_rd.0, primitive_wr.0, a, dt)
    }
}
