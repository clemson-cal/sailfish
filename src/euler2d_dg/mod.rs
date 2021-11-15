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

    fn euler2d_dg_limit_slopes(
        cell: node_2d::Cell,
        mesh: StructuredMesh,
        weights_rd_ptr: *const f64,
        weights_wr_ptr: *mut f64,
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

pub fn primitive_to_conserved(prim: &[f64], cons: &mut [f64], gamma_law_index: f64) {
    let rho = prim[0];
    let vx = prim[1];
    let vy = prim[2];
    let pre = prim[3];

    let px = vx * rho;
    let py = vy * rho;
    let kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    let thermal_energy = pre / (gamma_law_index - 1.0);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = kinetic_energy + thermal_energy;
}
