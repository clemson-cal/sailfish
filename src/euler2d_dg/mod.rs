use crate::node_2d;
use crate::{ExecutionMode, StructuredMesh};

pub mod solver;

extern "C" {

    fn euler2d_dg_advance_rk(
        cell: node_2d::Cell,
        mesh: StructuredMesh,
        weights_rd_ptr: *const f64,
        weights_wr_ptr: *mut f64,
        dt: f64,
        mode: ExecutionMode,
    );

    fn euler2d_dg_wavespeed(
        cell: node_2d::Cell,
        mesh: StructuredMesh,
        weights_ptr: *const f64,
        wavespeed_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn euler2d_dg_maximum(
        data: *const f64,
        size: std::os::raw::c_ulong,
        mode: ExecutionMode,
    ) -> f64;
    
}
