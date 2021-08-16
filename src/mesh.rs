use crate::IndexSpace;

#[derive(Clone, PartialOrd, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Mesh {
    Structured(crate::StructuredMesh),
    FacePositions1D(Vec<f64>),
}

impl Mesh {
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
