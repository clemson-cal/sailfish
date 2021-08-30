pub mod cmdline;
pub mod error;
pub mod euler1d;
pub mod euler2d;
pub mod iso2d;
pub mod sr1d;
pub mod lookup_table;
pub mod mesh;
pub mod parse;
pub mod patch;
pub mod setups;
pub mod state;
pub mod traits;

pub use crate::cmdline::CommandLine;
pub use crate::patch::Patch;
pub use crate::state::{Recurrence, RecurringTask, State};
pub use crate::traits::{PatchBasedBuild, PatchBasedSolve, Setup, Solve};
pub use gpu_core::Device;
pub use gridiron::index_space::IndexSpace;
pub use mesh::Mesh;

use cfg_if::cfg_if;
use std::ops::Range;
use std::str::FromStr;

/// Returns the current version number (should be consistent with Cargo
/// meta-data).
pub fn sailfish_version() -> String {
    "sailfish version 0.3.2".to_owned()
}

/// Execution modes. These modes are referenced by Rust driver code, and by
/// solver code written in C.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    /// Execution is either single core, or parallelized using thread-pool
    /// over a patch-based domain, without the help of OpenMP.
    CPU,
    /// Execution is parallelized in C code via OpenMP. If the domain is
    /// decomposed into patches, the patches are processed sequentially.
    OMP,
    /// Solver execution is performed on a GPU device, if available.
    GPU,
}

/// Description of sink model to model accretion onto a gravitating object.
/// 
/// C equivalent is defined in sailfish.h.
#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum SinkModel {
    /// No mass or momentum is subtracted around this point mass
    Inactive,
    /// The sink removes mass and momentum at the same rate so that the gas
    /// velocity is unchanged (most conventional)
    AccelerationFree,
    /// The sink does not change the fluid angular momentum, with respect to
    /// its position (most favorable)
    TorqueFree,
    /// The sink removes mass but not momentum (least favorable)
    ForceFree,
}

impl FromStr for SinkModel {
    type Err = error::Error;
    /// Tries to create a `SinkModel` from a string description. Returns
    /// an error if no match is found.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "inactive" | "none" => Ok(SinkModel::Inactive),
            "acceleration-free" | "af" => Ok(SinkModel::AccelerationFree),
            "torque-free" | "tf" => Ok(SinkModel::TorqueFree),
            "force-free" | "ff" => Ok(SinkModel::ForceFree),
            _ => Err(error::Error::UnknownEnumVariant {
                enum_type: "sink model".to_owned(),
                variant: s.to_owned(),
            }),
        }
    }
}

/// Description of basic equations of state supported by various solvers.
/// 
/// C equivalent is defined in sailfish.h.
/// 
/// Note: some solvers might be hard-coded to use a particular equation of
/// state.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum EquationOfState {
    Isothermal { sound_speed_squared: f64 },
    LocallyIsothermal { mach_number_squared: f64 },
    GammaLaw { gamma_law_index: f64 },
}

/// A gravitating point mass.
/// 
/// C equivalent is defined in sailfish.h.
#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct PointMass {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub mass: f64,
    pub rate: f64,
    pub radius: f64,
    pub model: SinkModel,
}

impl Default for PointMass {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            mass: 0.0,
            rate: 0.0,
            radius: 0.0,
            model: SinkModel::Inactive,
        }
    }
}

/// A fixed-length list of 0, 1, or 2 point masses.
/// 
/// C equivalent is defined in sailfish.h.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointMassList {
    masses: [PointMass; 2],
    count: i32,
}

impl PointMassList {
    pub fn from_slice(slice: &[PointMass]) -> Self {
        let mut masses = [PointMass::default(); 2];
        masses[..slice.len()].copy_from_slice(slice);
        Self {
            masses,
            count: slice.len() as i32,
        }
    }
    pub fn to_vec(self) -> Vec<PointMass> {
        self.masses[..self.count as usize].to_vec()
    }
}

impl Default for PointMassList {
    fn default() -> Self {
        Self::from_slice(&[])
    }
}

/// A description of a wave-damping (or buffer) zone to be used in
/// context-specific solver code.
/// 
/// C equivalent is defined in sailfish.h.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition {
    Default,
    Inflow,
    KeplerianBuffer {
        surface_density: f64,
        surface_pressure: f64,
        central_mass: f64,
        driving_rate: f64,
        outer_radius: f64,
        onset_width: f64,
    },
}

/// A logically cartesian 2d mesh with uniform grid spacing.
/// 
/// C equivalent is defined in sailfish.h.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StructuredMesh {
    /// Number of zones on the i-axis
    pub ni: i64,
    /// Number of zones on the j-axis
    pub nj: i64,
    /// Left coordinate edge of the domain
    pub x0: f64,
    /// Right coordinate edge of the domain
    pub y0: f64,
    /// Zone spacing on the i-axis
    pub dx: f64,
    /// Zone spacing on the j-axis
    pub dy: f64,
}

impl StructuredMesh {
    /// Creates a square mesh that is centered on the origin, with the given
    /// number of zones on each side.
    pub fn centered_square(domain_radius: f64, resolution: u32) -> Self {
        Self {
            x0: -domain_radius,
            y0: -domain_radius,
            ni: resolution as i64,
            nj: resolution as i64,
            dx: 2.0 * domain_radius / resolution as f64,
            dy: 2.0 * domain_radius / resolution as f64,
        }
    }

    /// Returns the number of zones on the i-axis as a `u32`.
    pub fn ni(&self) -> u32 {
        self.ni as u32
    }

    /// Returns the number of zones on the j-axis as a `u32`.
    pub fn nj(&self) -> u32 {
        self.nj as u32
    }

    /// Returns the number of total zones (`ni * nj`) as a `usize`.
    pub fn num_total_zones(&self) -> usize {
        (self.ni * self.nj) as usize
    }

    /// Returns the number of zones in each direction
    pub fn shape(&self) -> [u32; 2] {
        [self.ni as u32, self.nj as u32]
    }

    /// Returns the cell-center `[x, y]` coordinate at a given index.
    /// Out-of-bounds indexes are allowed.
    pub fn cell_coordinates(&self, i: i64, j: i64) -> [f64; 2] {
        let x = self.x0 + (i as f64 + 0.5) * self.dx;
        let y = self.y0 + (j as f64 + 0.5) * self.dy;
        [x, y]
    }

    /// Returns the vertex `[x, y]` coordinate at a given index. Out-of-bounds
    /// indexes are allowed.
    pub fn vertex_coordinates(&self, i: i64, j: i64) -> [f64; 2] {
        let x = self.x0 + i as f64 * self.dx;
        let y = self.y0 + j as f64 * self.dy;
        [x, y]
    }

    /// Returns a new structured mesh covering the given index range of this
    /// one.
    pub fn sub_mesh(&self, di: Range<i64>, dj: Range<i64>) -> Self {
        let [x0, y0] = self.vertex_coordinates(di.start, dj.start);
        let [ni, nj] = [di.count() as i64, dj.count() as i64];
        Self {
            ni,
            nj,
            x0,
            y0,
            dx: self.dx,
            dy: self.dy,
        }
    }
}

/// Describes a st of curvilinear coordinates to use.
/// 
/// C equivalent is defined in sailfish.h.
/// 
/// Note: some solvers might be hard-coded to use a particular coordinate
/// system.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Coordinates {
    Cartesian,
    SphericalPolar,
}

/// Returns whether the code has been compiled with OpenMP support,
/// `feature=omp`.
pub fn compiled_with_omp() -> bool {
    cfg_if! {
        if #[cfg(feature = "omp")] {
            true
        } else {
            false
        }
    }
}

/// Returns whether the code has been compiled with GPU support
/// (`feature=gpu`) either via CUDA or HIP.
pub fn compiled_with_gpu() -> bool {
    cfg_if! {
        if #[cfg(feature = "gpu")] {
            true
        } else {
            false
        }
    }
}
