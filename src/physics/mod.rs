use std::os::raw::c_void;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Number of zones on the i-axis
    pub ni: u32,
    /// Number of zones on the j-axis
    pub nj: u32,
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
    /// Returns the number of zones on the i-axis as a `usize`.
    pub fn ni(&self) -> usize {
        self.ni as usize
    }
    /// Returns the number of zones on the j-axis as a `usize`.
    pub fn nj(&self) -> usize {
        self.nj as usize
    }
    /// Returns the number of total zones (`ni * nj`) as a `usize`.
    pub fn num_total_zones(&self) -> usize {
        (self.ni * self.nj) as usize
    }
    /// Return the row-major memory strides. Assumes 3 conserved
    /// quantities.
    pub fn strides(&self) -> [usize; 2] {
        [self.nj as usize * 3, 3]
    }
}

#[repr(C)]
pub struct Solver(*mut c_void);

impl Solver {
    /// Creates a new solver instance from a mesh instance.
    pub fn new(mesh: Mesh) -> Self {
        unsafe { Self(solver_new(mesh.clone())) }
    }

    /// Retrieves a copy of the mesh struct used to create this solver
    /// instance.
    pub fn mesh(&self) -> Mesh {
        unsafe { solver_get_mesh(self.0) }
    }

    pub fn advance(&mut self, rk_order: u32, dt: f64) {
        match rk_order {
            1 => {
                self.advance_rk(0.0, dt);
            }
            2 => {
                self.new_timestep();
                self.advance_rk(0./1., dt);
                self.advance_rk(1./2., dt);
            }
            3 => {
                self.new_timestep();
                self.advance_rk(0./1., dt);
                self.advance_rk(3./4., dt);
                self.advance_rk(1./3., dt);
            }
            _ => {
                panic!("invalid runge-kutta order")
            }            
        }
    }

    /// Caches the conserved variables to the start of the time step to
    /// prepare for RK integration steps.
    fn new_timestep(&mut self) {
        unsafe { solver_new_timestep(self.0) }
    }

    fn advance_rk(&mut self, a: f64, dt: f64) {
        unsafe { solver_advance_rk(self.0, a, dt) }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        unsafe { solver_del(self.0) }
    }
}

extern "C" {
    pub(crate) fn solver_new(mesh: Mesh) -> *mut c_void;
    pub(crate) fn solver_del(solver: *mut c_void);
    pub(crate) fn solver_get_mesh(solver: *mut c_void) -> Mesh;
    pub(crate) fn solver_new_timestep(solver: *mut c_void);
    pub(crate) fn solver_advance_rk(solver: *mut c_void, a: f64, dt: f64);
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum EquationOfState {
    Isothermal { sound_speed: f64 },
    LocallyIsothermal { mach_number: f64 },
    GammaLaw { gamma_law_index: f64 },
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointMass {
    pub x: f64,
    pub y: f64,
    pub mass: f64,
    pub rate: f64,
    pub radius: f64,
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
    }
}
