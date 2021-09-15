use crate::{EquationOfState, ExecutionMode, StructuredMesh};

pub mod solver;
pub mod node;

extern "C" {
    fn euler2d_dg_primitive_to_weights(
        cell: node::Cell,
        mesh: StructuredMesh,
        primitive_ptr: *const f64,
        weights_ptr: *mut f64,
        mode: ExecutionMode,
    );

    fn euler2d_dg_advance_rk(
        cell: node::Cell,
        mesh: StructuredMesh,
        weights_rd_ptr: *const f64,
        weights_wr_ptr: *mut f64,
        eos: EquationOfState,
        dt: f64,
        mode: ExecutionMode,
    );
}
