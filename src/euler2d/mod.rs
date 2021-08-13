//use crate::error::Error;
use crate::sailfish::{
    BufferZone, EquationOfState, ExecutionMode, PointMass, StructuredMesh, //Solve
};
//use crate::Setup;
//use cfg_if::cfg_if;

pub mod solver;

extern "C" {
    pub fn euler2d_primitive_to_conserved(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn euler2d_advance_rk(
        mesh: StructuredMesh,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        eos: EquationOfState,
        buffer: BufferZone,
        masses: *const PointMass,
        num_masses: i32,
        alpha: f64,
        a: f64,
        dt: f64,
        velocity_ceiling: f64,
        cooling_coefficient: f64,
        mach_ceiling: f64,
        density_floor: f64,
        pressure_floor: f64,
        mode: ExecutionMode,
    );

    pub fn euler2d_wavespeed(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        eos: EquationOfState,
        mode: ExecutionMode,
    );

    pub fn euler2d_maximum(
        data: *const f64,
        size: std::os::raw::c_ulong,
        mode: ExecutionMode
    ) -> f64;
}
