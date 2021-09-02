pub mod solver;
pub mod node;

extern "C" {
    pub fn iso2d_dg_say_hello(order: i32) -> i32;
    pub fn iso2d_dg_get_order(cell: node::Cell) -> i32;
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
