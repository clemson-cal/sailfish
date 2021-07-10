#[repr(C)]
#[derive(Clone, Copy)]
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
    None,
    Keplerian {
        surface_density: f64,
        central_mass: f64,
        driving_rate: f64,
        outer_radius: f64,
        onset_width: f64,
    },
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Mesh {
    /// Number of zones on the i-axis
    pub ni: i32,
    /// Number of zones on the j-axis
    pub nj: i32,
    /// Left coordinate edge of the domain
    pub x0: f64,
    /// Right coordinate edge of the domain
    pub y0: f64,
    /// Zone spacing on the i-axis
    pub dx: f64,
    /// Zone spacing on the j-axis
    pub dy: f64,
}

impl Mesh {
    /// Creates a square mesh that is centered on the origin, with the given
    /// number of zones on each side.
    pub fn centered_square(domain_radius: f64, resolution: u32) -> Self {
        Self {
            x0: -domain_radius,
            y0: -domain_radius,
            ni: resolution as i32,
            nj: resolution as i32,
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
    /// Returns the cell-center [x, y] coordinate at a given index.
    /// Out-of-bounds indexes are allowed.
    pub fn cell_coordinates(&self, i: i32, j: i32) -> [f64; 2] {
        let x = self.x0 + (i as f64 + 0.5) * self.dx;
        let y = self.y0 + (j as f64 + 0.5) * self.dy;
        [x, y]
    }
}

pub trait Solve {
    /// Returns the primitive variable array for this solver. The data is
    /// row-major with contiguous primitive variable components. The array
    /// includes guard zones.
    fn primitive(&self) -> Vec<f64>;

    /// Converts the internal primitive variable array to a conserved variable
    /// array, and stores that array in the solver's conserved variable buffer.
    fn primitive_to_conserved(&mut self);

    /// Advances the primitive variable array by one low-storage Runge-Kutta
    /// sub-stup.
    #[allow(clippy::too_many_arguments)]
    fn advance_rk(
        &mut self,
        nu: f64,
        eos: EquationOfState,
        buffer: BufferZone,
        masses: &[PointMass],
        a: f64,
        dt: f64,
        velocity_ceiling: f64,
    );

    /// Returns the largest wavespeed among the zones in the solver's current
    /// primitive array.
    fn max_wavespeed(&self, eos: EquationOfState, masses: &[PointMass]) -> f64;
}
