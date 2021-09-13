use crate::{ExecutionMode, StructuredMesh};

pub mod solver;

extern "C" {
    pub fn euler_rz_primitive_to_conserved(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn euler_rz_advance_rk(
        mesh: StructuredMesh,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        a: f64,
        dt: f64,
        mode: ExecutionMode,
    );

    pub fn euler_rz_wavespeed(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn euler_rz_maximum(
        data: *const f64,
        size: std::os::raw::c_ulong,
        mode: ExecutionMode,
    ) -> f64;
}
