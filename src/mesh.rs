//! Generalizes over mesh data structures used by different solvers.

use crate::{IndexSpace, StructuredMesh};

/// Either a [`crate::StructuredMesh`] or a `Vec` of face positions in 1D.
#[derive(Clone, PartialOrd, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Mesh {
    Structured(crate::StructuredMesh),
    FacePositions1D(Vec<f64>),
}

impl Mesh {
    /// Creates a square mesh that is centered on the origin, with the given
    /// number of zones on each side.
    pub fn centered_square(domain_radius: f64, resolution: u32) -> Self {
        Self::Structured(StructuredMesh::centered_square(domain_radius, resolution))
    }

    /// Creates a `FacePositions1D` mesh with logarithmic zone spacing.
    pub fn logarithmic_radial(inner_radius: f64, num_decades: u32, zones_per_decade: u32) -> Self {
        let faces = (0..(zones_per_decade * num_decades + 1) as u32)
            .map(|i| inner_radius * f64::powf(10.0, i as f64 / zones_per_decade as f64))
            .collect();
        Self::FacePositions1D(faces)
    }

    pub fn num_total_zones(&self) -> usize {
        match self {
            Self::Structured(mesh) => mesh.num_total_zones(),
            Self::FacePositions1D(faces) => faces.len() - 1,
        }
    }

    pub fn min_spacing(&self) -> f64 {
        match self {
            Self::Structured(mesh) => f64::min(mesh.dx, mesh.dy),
            Self::FacePositions1D(faces) => {
                faces.windows(2).map(|w| w[1] - w[0]).fold(f64::MAX, f64::min)
            }
        }
    }

    pub fn index_space(&self) -> IndexSpace {
        match self {
            Self::Structured(mesh) => IndexSpace::new(0..mesh.ni, 0..mesh.nj),
            Self::FacePositions1D(faces) => IndexSpace::new(0..faces.len() as i64 - 1, 0..1),
        }
    }
}
