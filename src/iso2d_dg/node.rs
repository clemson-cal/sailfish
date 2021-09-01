#![allow(unused)]
//! There are interior nodes and face nodes (interior quadrature points or face
//! quadrature points)
//!
//! The face dimensionality is one smaller than the interior
//!
//! NI := NUM_INTERIOR_NODES
//! NF := NUM_FACE_NODES
//! NP := NUM_POLYNOMIALS

#[derive(Clone, Copy)]
pub struct NodeData<const NP: usize> {
    /// normalized cell x coordinate [-1, 1]
    xsi_x: f64,
    /// normalized cell y coordinate [-1, 1]
    xsi_y: f64,
    /// cell basis functions phi(xsi_x, xsi_y)
    phi: [f64; NP],
    /// xsi_x derivatives of basis functions
    dphi_dx: [f64; NP],
    /// xsi_y derivatives of basis functions
    dphi_dy: [f64; NP],
    /// Guassian weight, multiplied by n-hat, for face nodes
    weight: f64,
}

impl<const NP: usize> NodeData<NP> {
    fn new(n: usize, m: usize, x: f64, y: f64, weight: f64) -> Self {
        let mut node = Self::default();
        node.xsi_x = x;
        node.xsi_y = y;
        node.weight = weight;
        for p in 0..NP {
            node.phi[p] = pn(n, x) * pn(m, y);
            node.dphi_dx[p] = pn_prime(n, x) * pn(m, y);
            node.dphi_dy[p] = pn(n, x) * pn_prime(m, y);
        }
        node
    }
}

impl<const NP: usize> Default for NodeData<NP> {
    fn default() -> Self {
        Self {
            xsi_x: 0.0,
            xsi_y: 0.0,
            phi: [0.0; NP],
            dphi_dx: [0.0; NP],
            dphi_dy: [0.0; NP],
            weight: 0.0,
        }
    }
}

pub struct Cell<const NI: usize, const NF: usize, const NP: usize> {
    interior_nodes: [NodeData<NP>; NI],
    face_nodes_li: [NodeData<NP>; NF],
    face_nodes_ri: [NodeData<NP>; NF],
    face_nodes_lj: [NodeData<NP>; NF],
    face_nodes_rj: [NodeData<NP>; NF],
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

impl<const NI: usize, const NF: usize, const NP: usize> Cell<NI, NF, NP> {
    pub fn new() -> Self {
        let order = NF;
        let x = gaussian_quadrature_points(order);
        let w = gaussian_weight(order);

        let mut interior_nodes = [NodeData::<NP>::default(); NI];
        let mut face_nodes_li = [NodeData::<NP>::default(); NF];
        let mut face_nodes_ri = [NodeData::<NP>::default(); NF];
        let mut face_nodes_lj = [NodeData::<NP>::default(); NF];
        let mut face_nodes_rj = [NodeData::<NP>::default(); NF];

        let mut q = 0;
        for i in 0..NF {
            for j in 0..NF {
                for n in 0..order {
                    for m in 0..n + 1 {
                        interior_nodes[q] = NodeData::new(n, m, x[i], x[j], w[i] * w[j]);
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
                for j in 0..NF {
                    face_nodes_li[q] = NodeData::new(n, m, l, x[j], l * w[j]);
                    face_nodes_ri[q] = NodeData::new(n, m, r, x[j], r * w[j]);
                }
                for i in 0..NF {
                    face_nodes_lj[q] = NodeData::new(n, m, x[i], l, w[i] * l);
                    face_nodes_rj[q] = NodeData::new(n, m, x[i], r, w[i] * r);                    
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

// type Cell1 = Cell<1, 1, 1>;
// type Cell2 = Cell<4, 2, 3>;
// type Cell3 = Cell<9, 3, 6>;
// type Cell4 = Cell<16, 4, 10>;
// type Cell5 = Cell<25, 5, 15>;
