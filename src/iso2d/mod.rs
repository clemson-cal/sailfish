use crate::sailfish::{BufferZone, EquationOfState, ExecutionMode, Mesh, PointMass};

extern "C" {
    pub fn iso2d_primitive_to_conserved(
        mesh: Mesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn iso2d_advance_rk(
        mesh: Mesh,
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
        mode: ExecutionMode,
    );

    pub fn iso2d_wavespeed(
        mesh: Mesh,
        primitive_ptr: f64,
        wavespeed_ptr: f64,
        eos: EquationOfState,
        masses: *const PointMass,
        num_masses: i32,
        mode: ExecutionMode,
    );
}

pub struct Solver {}
