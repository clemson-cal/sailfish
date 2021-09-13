use crate::{EquationOfState, ExecutionMode, StructuredMesh};

pub mod solver;
pub mod node;

extern "C" {

    pub fn iso2d_dg_primitive_to_weights(
        cell: node::Cell,
        mesh: StructuredMesh,
        weights_ptr: *const f64,
        weights_ptr: *mut f64,
        mode: ExecutionMode,
    );

    pub fn iso2d_dg_advance_rk(
        cell: node::Cell,
        mesh: StructuredMesh,
        weights_rd_ptr: *const f64,
        weights_wr_ptr: *mut f64,
        eos: EquationOfState,
        dt: f64,
        mode: ExecutionMode,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_call_into_c_api() {
        unsafe {
            assert_eq!(iso2d_dg_say_hello(5), 5);
        }
    }

    #[test]
    fn can_get_cell_order_from_c() {
        let cell = node::Cell::new(5);
        unsafe {
            assert_eq!(iso2d_dg_get_order(cell), 5);
        }
    }
}
