use std::ops::Range;
use crate::Setup;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    CPU,
    OMP,
    GPU,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum SinkModel {
    Inactive,
    AccelerationFree,
    TorqueFree,
    ForceFree,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum EquationOfState {
    Isothermal { sound_speed_squared: f64 },
    LocallyIsothermal { mach_number_squared: f64 },
    GammaLaw { gamma_law_index: f64 },
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum BufferZone {
    NoBuffer,
    Keplerian {
        surface_density: f64,
        central_mass: f64,
        driving_rate: f64,
        outer_radius: f64,
        onset_width: f64,
    },
}

/// A logically cartesian 2d mesh with uniform grid spacing. C equivalent is
/// defined in sailfish.h.
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
    pub fn subset_mesh(&self, di: Range<i64>, dj: Range<i64>) -> Self {
        let [x0, y0] = self.vertex_coordinates(di.start, dj.start);
        let [ni, nj] = [di.count() as i64, dj.count() as i64];
        Self {
            ni, nj, x0, y0, dx: self.dx, dy: self.dy,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Coordinates {
    Cartesian,
    SphericalPolar,
}

pub trait Solve {
    /// Returns the primitive variable array for this solver. The data is
    /// row-major with contiguous primitive variable components. The array
    /// includes guard zones.
    fn primitive(&self) -> Vec<f64>;

    /// Converts the internal primitive variable array to a conserved variable
    /// array, and stores that array in the solver's conserved variable buffer.
    fn primitive_to_conserved(&mut self);

    /// Returns the largest wavespeed among the zones in the solver's current
    /// primitive array.
    fn max_wavespeed(&self, time: f64, setup: &dyn Setup) -> f64;

    /// Advances the primitive variable array by one low-storage Runge-Kutta
    /// sub-stup.
    fn advance_rk(
        &mut self,
        setup: &dyn Setup,
        time: f64,
        a: f64,
        dt: f64,
        velocity_ceiling: f64,
    );

    /// Primitive variable array in a solver using first, second, or third-order
    /// Runge-Kutta time stepping.
    fn advance(
        &mut self,
        setup: &dyn Setup,
        rk_order: u32,
        time: f64,
        dt: f64,
        velocity_ceiling: f64,
    ) {
        self.primitive_to_conserved();
        match rk_order {
            1 => {
                self.advance_rk(setup, time, 0.0, dt, velocity_ceiling);
            }
            2 => {
                self.advance_rk(setup, time + 0.0 * dt, 0.0, dt, velocity_ceiling);
                self.advance_rk(setup, time + 1.0 * dt, 0.5, dt, velocity_ceiling);
            }
            3 => {
                // t1 = a1 * tn + (1 - a1) * (tn + dt) =     tn +     (      dt) = tn +     dt [a1 = 0]
                // t2 = a2 * tn + (1 - a2) * (t1 + dt) = 3/4 tn + 1/4 (tn + 2dt) = tn + 1/2 dt [a2 = 3/4]
                self.advance_rk(setup, time + 0.0 * dt, 0. / 1., dt, velocity_ceiling);
                self.advance_rk(setup, time + 1.0 * dt, 3. / 4., dt, velocity_ceiling);
                self.advance_rk(setup, time + 0.5 * dt, 1. / 3., dt, velocity_ceiling);
            }
            _ => {
                panic!("invalid RK order")
            }
        }
    }
}
