use crate::sailfish;

#[derive(Clone, PartialOrd, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Mesh {
    Structured(sailfish::StructuredMesh),
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
            Self::FacePositions1D(_faces) => {
                todo!()
            }
        }        
    }
}
