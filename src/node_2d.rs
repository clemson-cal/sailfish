// For reference --- num interior nodes, num face nodes, and num polynomials,
// for DG orders 1 through 5:
//
// 1, 1, 1
// 4, 2, 3
// 9, 3, 6
// 16, 4, 10
// 25, 5, 15
const MAX_INTERIOR_NODES: usize = 9;
const MAX_FACE_NODES: usize = 3;
const MAX_POLYNOMIALS: usize = 6;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NodeData {
    /// normalized cell x coordinate [-1, 1]
    xsi_x: f64,
    /// normalized cell y coordinate [-1, 1]
    xsi_y: f64,
    /// cell basis functions phi(xsi_x, xsi_y)
    phi: [f64; MAX_POLYNOMIALS],
    /// xsi_x derivatives of basis functions
    dphi_dx: [f64; MAX_POLYNOMIALS],
    /// xsi_y derivatives of basis functions
    dphi_dy: [f64; MAX_POLYNOMIALS],
    /// Guassian weight, multiplied by n-hat, for face nodes
    weight: f64,
}

impl NodeData {
    fn new(order: usize, x: f64, y: f64, weight: f64) -> Self {
        let mut node = Self::default();
        node.xsi_x = x;
        node.xsi_y = y;
        node.weight = weight;
        let mut p = 0;
        for n in 0..order {
            for m in 0..order {
                if n + m < order {
                    node.phi[p] = pn(n, x) * pn(m, y);
                    node.dphi_dx[p] = pn_prime(n, x) * pn(m, y);
                    node.dphi_dy[p] = pn(n, x) * pn_prime(m, y);
                    p += 1;
                }
            }
        }
        node
    }

    fn describe(&self, order: usize) -> String {
        use std::fmt::Write;
        let mut d = String::new();
        writeln!(
            d,
            "weight = {}; (x y) = ({} {})",
            self.weight, self.xsi_x, self.xsi_y
        )
        .unwrap();
        write!(d, "phi: [ ").unwrap();
        for n in 0..num_polynomials(order) {
            write!(d, "{} ", self.phi[n]).unwrap()
        }
        writeln!(d, "]").unwrap();
        write!(d, "dphi_dx: [ ").unwrap();
        for n in 0..num_polynomials(order) {
            write!(d, "{} ", self.dphi_dx[n]).unwrap()
        }
        writeln!(d, "]").unwrap();
        write!(d, "dphi_dy: [ ").unwrap();
        for n in 0..num_polynomials(order) {
            write!(d, "{} ", self.dphi_dy[n]).unwrap()
        }
        writeln!(d, "]").unwrap();
        d
    }
}

impl Default for NodeData {
    fn default() -> Self {
        Self {
            xsi_x: 0.0,
            xsi_y: 0.0,
            phi: [0.0; MAX_POLYNOMIALS],
            dphi_dx: [0.0; MAX_POLYNOMIALS],
            dphi_dy: [0.0; MAX_POLYNOMIALS],
            weight: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Cell {
    pub interior_nodes: [NodeData; MAX_INTERIOR_NODES],
    pub face_nodes_li: [NodeData; MAX_FACE_NODES],
    pub face_nodes_ri: [NodeData; MAX_FACE_NODES],
    pub face_nodes_lj: [NodeData; MAX_FACE_NODES],
    pub face_nodes_rj: [NodeData; MAX_FACE_NODES],
    order: i32,
}

impl Cell {
    pub fn new(order: usize) -> Self {
        let x = gaussian_quadrature_points(order);
        let w = gaussian_weight(order);

        let mut interior_nodes = [NodeData::default(); MAX_INTERIOR_NODES];
        let mut face_nodes_li = [NodeData::default(); MAX_FACE_NODES];
        let mut face_nodes_ri = [NodeData::default(); MAX_FACE_NODES];
        let mut face_nodes_lj = [NodeData::default(); MAX_FACE_NODES];
        let mut face_nodes_rj = [NodeData::default(); MAX_FACE_NODES];

        let mut q = 0;
        for i in 0..order {
            for j in 0..order {
                interior_nodes[q] = NodeData::new(order, x[i], x[j], w[i] * w[j]);
                q += 1;
            }
        }

        // nl and nr are unit normal vectors of the left and right faces,
        // dotted with the coordinate direction. For face nodes, the Gaussian
        // weight is multiplied by this quantity for computing sums since
        // neither is required independently.
        let [nl, nr] = [-1.0, 1.0];

        // xl and xr are the positions of the left and right zone faces in
        // local zone coordinates, elsewhere referred to as xsi.
        let [xl, xr] = [-1.0, 1.0];

        for j in 0..order {
            face_nodes_li[j] = NodeData::new(order, xl, x[j], nl * w[j]);
            face_nodes_ri[j] = NodeData::new(order, xr, x[j], nr * w[j]);
        }
        for i in 0..order {
            face_nodes_lj[i] = NodeData::new(order, x[i], xl, w[i] * nl);
            face_nodes_rj[i] = NodeData::new(order, x[i], xr, w[i] * nr);
        }

        Self {
            interior_nodes,
            face_nodes_li,
            face_nodes_ri,
            face_nodes_lj,
            face_nodes_rj,
            order: order as i32,
        }
    }

    pub fn order(&self) -> usize {
        self.order as usize
    }

    pub fn interior_nodes(&self) -> impl Iterator<Item = &NodeData> + '_ {
        self.interior_nodes
            .iter()
            .take(self.num_quadrature_points())
    }

    pub fn quadrature_points(&self) -> impl Iterator<Item = [f64; 2]> + '_ {
        self.interior_nodes().map(|node| [node.xsi_x, node.xsi_y])
    }

    pub fn num_quadrature_points(&self) -> usize {
        self.order() * self.order()
    }

    pub fn num_polynomials(&self) -> usize {
        num_polynomials(self.order())
    }

    /// Converts a multi-component scalar field `y`, tabulated at the
    /// quadrature points of this cell, to a component-wise weights
    /// representation. The size of the input slice `y` is the number of
    /// quadrature points in this cell, times the number of fields
    /// `num_scalars`. The size of the output slice `weights` is the number of
    /// polynomial basis functions for this cell, times `num_scalars`. The
    /// scalar field components are contiguous in memory at each quadtrature
    /// point.
    pub fn to_weights(&self, y: &[f64], weights: &mut [f64], num_scalars: usize) {
        let np = self.num_polynomials();

        assert_eq!(y.len(), num_scalars * self.num_quadrature_points());
        assert_eq!(weights.len(), num_scalars * np);

        for w in weights.iter_mut() {
            *w = 0.0;
        }

        for (node, yp) in self.interior_nodes().zip(y.chunks_exact(num_scalars)) {
            for q in 0..num_scalars {
                for l in 0..self.num_polynomials() {
                    weights[q * np + l] += 0.25 * yp[q] * node.phi[l] * node.weight;
                }
            }
        }
    }

    pub fn describe(&self) -> String {
        use std::fmt::Write;
        let o = self.order();
        let mut d = String::new();
        writeln!(d, "cell order: {}", self.order).unwrap();
        for (n, node) in self.interior_nodes().enumerate() {
            writeln!(d, "interior node {}: {}", n, node.describe(o)).unwrap();
        }
        for (n, node) in self.face_nodes_li.iter().take(o).enumerate() {
            writeln!(d, "face node li {}: {}", n, node.describe(o)).unwrap();
        }
        for (n, node) in self.face_nodes_ri.iter().take(o).enumerate() {
            writeln!(d, "face node ri {}: {}", n, node.describe(o)).unwrap();
        }
        for (n, node) in self.face_nodes_lj.iter().take(o).enumerate() {
            writeln!(d, "face node lj {}: {}", n, node.describe(o)).unwrap();
        }
        for (n, node) in self.face_nodes_rj.iter().take(o).enumerate() {
            writeln!(d, "face node rj {}: {}", n, node.describe(o)).unwrap();
        }
        d
    }
}

/// Returns the Legendre polynomials, for `n = 0` through 4, scaled by
/// `sqrt(2n + 1)`.
fn pn(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => f64::sqrt(3.0) * x,
        2 => f64::sqrt(5.0) * 0.500 * (3.0 * x * x - 1.0),
        3 => f64::sqrt(7.0) * 0.500 * (5.0 * x * x * x - 3.0 * x),
        4 => f64::sqrt(9.0) * 0.125 * (35.0 * x * x * x * x - 30.0 * x * x + 3.0),
        _ => panic!("unsupported pn order"),
    }
}

/// Returns the derivative of Legendre polynomials, for `n = 0` through 4,
/// scaled by `sqrt(2n + 1)`.
fn pn_prime(n: usize, x: f64) -> f64 {
    match n {
        0 => 0.0,
        1 => f64::sqrt(3.0),
        2 => f64::sqrt(5.0) * 0.500 * (6.0 * x),
        3 => f64::sqrt(7.0) * 0.500 * (15.0 * x * x - 3.0),
        4 => f64::sqrt(9.0) * 0.125 * (140.0 * x * x * x - 60.0 * x),
        _ => panic!("unsupported pn_prime order"),
    }
}

fn gaussian_quadrature_points(order: usize) -> [f64; 5] {
    match order {
        1 => [0.0, 0.0, 0.0, 0.0, 0.0],
        2 => [-1.0 / f64::sqrt(3.0), 1.0 / f64::sqrt(3.0), 0.0, 0.0, 0.0],
        3 => [-f64::sqrt(3.0 / 5.0), 0.0, f64::sqrt(3.0 / 5.0), 0.0, 0.0],
        4 => [
            f64::sqrt(3.0 / 7.0 + 2.0 / 7.0 * f64::sqrt(6.0 / 5.0)) * -1.0,
            f64::sqrt(3.0 / 7.0 - 2.0 / 7.0 * f64::sqrt(6.0 / 5.0)) * -1.0,
            f64::sqrt(3.0 / 7.0 - 2.0 / 7.0 * f64::sqrt(6.0 / 5.0)),
            f64::sqrt(3.0 / 7.0 + 2.0 / 7.0 * f64::sqrt(6.0 / 5.0)),
            0.0,
        ],
        5 => [
            1.0 / 3.0 * f64::sqrt(5.0 + 2.0 * f64::sqrt(10.0 / 7.0)) * -1.0,
            1.0 / 3.0 * f64::sqrt(5.0 - 2.0 * f64::sqrt(10.0 / 7.0)) * -1.0,
            0.0,
            1.0 / 3.0 * f64::sqrt(5.0 - 2.0 * f64::sqrt(10.0 / 7.0)),
            1.0 / 3.0 * f64::sqrt(5.0 + 2.0 * f64::sqrt(10.0 / 7.0)),
        ],
        _ => panic!("unsupported order"),
    }
}

fn gaussian_weight(order: usize) -> [f64; 5] {
    match order {
        1 => [2.0, 0.0, 0.0, 0.0, 0.0],
        2 => [1.0, 1.0, 0.0, 0.0, 0.0],
        3 => [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0, 0.0, 0.0],
        4 => [
            (18.0 - f64::sqrt(30.0)) / 36.0,
            (18.0 + f64::sqrt(30.0)) / 36.0,
            (18.0 + f64::sqrt(30.0)) / 36.0,
            (18.0 - f64::sqrt(30.0)) / 36.0,
            0.0,
        ],
        5 => [
            (322.0 - 13.0 * f64::sqrt(70.0)) / 900.0,
            (322.0 + 13.0 * f64::sqrt(70.0)) / 900.0,
            128.0 / 225.0,
            (322.0 + 13.0 * f64::sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * f64::sqrt(70.0)) / 900.0,
        ],
        _ => panic!("unsupported order"),
    }
}

fn num_polynomials(order: usize) -> usize {
    match order {
        1 => 1,
        2 => 3,
        3 => 6,
        4 => 10,
        5 => 15,
        _ => panic!(),
    }
}
