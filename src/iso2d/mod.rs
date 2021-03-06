use crate::{
    BoundaryCondition, EquationOfState, ExecutionMode, PointMass, PointMassList, StructuredMesh,
};

pub mod solver;

extern "C" {
    pub fn iso2d_primitive_to_conserved(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        conserved_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn iso2d_advance_rk(
        mesh: StructuredMesh,
        conserved_rk_ptr: *const f64,
        primitive_rd_ptr: *const f64,
        primitive_wr_ptr: *mut f64,
        eos: EquationOfState,
        boundary_condition: BoundaryCondition,
        mass_list: PointMassList,
        nu: f64,
        a: f64,
        dt: f64,
        velocity_ceiling: f64,
        mode: ExecutionMode,
    );

    pub fn iso2d_point_mass_source_term(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        cons_rate_ptr: *const f64,
        mass: PointMass,
        mode: ExecutionMode,
    );

    pub fn iso2d_wavespeed(
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        eos: EquationOfState,
        mass_list: PointMassList,
        mode: ExecutionMode,
    );

    pub fn iso2d_maximum(data: *const f64, size: std::os::raw::c_ulong, mode: ExecutionMode)
        -> f64;
}
