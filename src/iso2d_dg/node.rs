//! There are interior nodes and face nodes (interior quadrature points or face
//! quadrature points).
//!
//! The face dimensionality is one smaller than the interior.
//!
//! `NI := NUM_INTERIOR_NODES`
//! `NF := NUM_FACE_NODES`
//! `NP := NUM_POLYNOMIALS`
//!
//! `type Cell1 = Cell<1, 1, 1>;`
//! `type Cell2 = Cell<4, 2, 3>;`
//! `type Cell3 = Cell<9, 3, 6>;`
//! `type Cell4 = Cell<16, 4, 10>;`
//! `type Cell5 = Cell<25, 5, 15>;`

const MAX_INTERIOR_NODES: usize = 25;
const MAX_FACE_NODES: usize = 5;
const MAX_POLYNOMIALS: usize = 15;

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
    fn new(n: usize, m: usize, x: f64, y: f64, weight: f64, order: usize) -> Self {
        let mut node = Self::default();
        let num_polynomials = match order {
            1 => 1,
            2 => 3,
            3 => 6,
            4 => 10,
            5 => 15,
            _ => panic!("unsupported order"),
        };
        node.xsi_x = x;
        node.xsi_y = y;
        node.weight = weight;
        for p in 0..num_polynomials {
            node.phi[p] = pn(n, x) * pn(m, y);
            node.dphi_dx[p] = pn_prime(n, x) * pn(m, y);
            node.dphi_dy[p] = pn(n, x) * pn_prime(m, y);
        }
        node
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

#[derive(Clone, Copy)]
pub struct Cell {
    pub interior_nodes: [NodeData; MAX_INTERIOR_NODES],
    pub face_nodes_li: [NodeData; MAX_FACE_NODES],
    pub face_nodes_ri: [NodeData; MAX_FACE_NODES],
    pub face_nodes_lj: [NodeData; MAX_FACE_NODES],
    pub face_nodes_rj: [NodeData; MAX_FACE_NODES],
}

impl Cell {
    pub fn new(order: usize) -> Self {
        let num_faces = order;
        let x = gaussian_quadrature_points(order);
        let w = gaussian_weight(order);

        let mut interior_nodes = [NodeData::default(); MAX_INTERIOR_NODES];
        let mut face_nodes_li = [NodeData::default(); MAX_FACE_NODES];
        let mut face_nodes_ri = [NodeData::default(); MAX_FACE_NODES];
        let mut face_nodes_lj = [NodeData::default(); MAX_FACE_NODES];
        let mut face_nodes_rj = [NodeData::default(); MAX_FACE_NODES];

        let mut q = 0;
        for i in 0..num_faces {
            for j in 0..num_faces {
                for n in 0..order {
                    for m in 0..n + 1 {
                        interior_nodes[q] = NodeData::new(n, m, x[i], x[j], w[i] * w[j], order);
                    }
                }
                q += 1;
            }
        }

        let mut q = 0;
        for n in 0..order {
            for m in 0..n + 1 {
                let l = -1.0;
                let r = 1.0;
                for j in 0..num_faces {
                    face_nodes_li[q] = NodeData::new(n, m, l, x[j], l * w[j], order);
                    face_nodes_ri[q] = NodeData::new(n, m, r, x[j], r * w[j], order);
                }
                for i in 0..num_faces {
                    face_nodes_lj[q] = NodeData::new(n, m, x[i], l, w[i] * l, order);
                    face_nodes_rj[q] = NodeData::new(n, m, x[i], r, w[i] * r, order);                    
                }
                q += 1;
            }
        }

        Self {
            interior_nodes,
            face_nodes_li,
            face_nodes_ri,
            face_nodes_lj,
            face_nodes_rj,
        }
    }
}

fn pn(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => f64::sqrt(3.0) * x,
        2 => f64::sqrt(5.0) * 0.5 * (3.0 * x * x - 1.0),
        3 => f64::sqrt(7.0) * 0.5 * (5.0 * x * x * x - 3.0 * x),
        4 => 3.0 / 8.0 * (35.0 * x * x * x * x - 30.0 * x * x + 3.0),
        _ => panic!("unsupported order"),
    }
}

fn pn_prime(n: usize, x: f64) -> f64 {
    match n {
        0 => 0.0, // 1.0,
        1 => f64::sqrt(3.0),
        2 => f64::sqrt(5.0) * 0.5 * (6.0 * x),
        3 => f64::sqrt(7.0) * 0.5 * (15.0 * x * x - 3.0),
        4 => 3.0 / 8.0 * (140.0 * x * x * x - 60.0 * x),
        _ => panic!("unsupported order"),
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
