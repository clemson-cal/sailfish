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

#[derive(Clone, Copy)]
pub enum ExecutionMode {
    CPU,
    OMP,
    GPU,
}

#[repr(C)]
pub struct Solver(*mut c_void, ExecutionMode);

impl Solver {
    /// Creates a new solver instance from a mesh instance.
    pub fn new(mesh: Mesh, mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::CPU => Self(unsafe { solver_new_cpu(mesh) }, mode),
            ExecutionMode::OMP => Self(unsafe { solver_new_omp(mesh) }, mode),
            ExecutionMode::GPU => Self(unsafe { solver_new_gpu(mesh) }, mode),
        }
    }

    /// Retrieves a copy of the mesh struct used to create this solver
    /// instance.
    pub fn mesh(&self) -> Mesh {
        match self.1 {
            ExecutionMode::CPU => unsafe { solver_get_mesh_cpu(self.0) },
            ExecutionMode::OMP => unsafe { solver_get_mesh_omp(self.0) },
            ExecutionMode::GPU => unsafe { solver_get_mesh_gpu(self.0) },
        }
    }

    /// Retrieves a copy of the primitive variable array.
    pub fn primitive(&self) -> Vec<f64> {
        let mut primitive = vec![0.0; self.mesh().num_total_zones() * 3];
        match self.1 {
            ExecutionMode::CPU => unsafe { solver_get_primitive_cpu(self.0, primitive.as_mut_ptr()) }
            ExecutionMode::OMP => unsafe { solver_get_primitive_omp(self.0, primitive.as_mut_ptr()) }
            ExecutionMode::GPU => unsafe { solver_get_primitive_gpu(self.0, primitive.as_mut_ptr()) }
        }
        primitive
    }

    /// Sets the primitive variable array. The data is row-major (the last
    /// index increases fastest), and each element must contain the same
    /// number of doubles as there are primitive quantities.
    pub fn set_primitive(&self, primitive: &[f64]) {
        if primitive.len() != self.mesh().num_total_zones() * 3 {
            panic!("wrong number of zones for primitive array")
        }
        match self.1 {
            ExecutionMode::CPU => unsafe { solver_set_primitive_cpu(self.0, primitive.as_ptr()) }
            ExecutionMode::OMP => unsafe { solver_set_primitive_omp(self.0, primitive.as_ptr()) }
            ExecutionMode::GPU => unsafe { solver_set_primitive_gpu(self.0, primitive.as_ptr()) }
        }
    }

    /// Advances the solution by the given time step, using a low-storage
    /// Runge-Kutta of the given order (1, 2, or 3).
    pub fn advance(&mut self, rk_order: u32, dt: f64) {
        match rk_order {
            1 => {
                self.new_timestep();
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
        match self.1 {
            ExecutionMode::CPU => unsafe { solver_new_timestep_cpu(self.0) }
            ExecutionMode::OMP => unsafe { solver_new_timestep_omp(self.0) }
            ExecutionMode::GPU => unsafe { solver_new_timestep_gpu(self.0) }
        }
    }

    fn advance_rk(&mut self, a: f64, dt: f64) {
        match self.1 {
            ExecutionMode::CPU => unsafe { solver_advance_rk_cpu(self.0, a, dt) }
            ExecutionMode::OMP => unsafe { solver_advance_rk_omp(self.0, a, dt) }
            ExecutionMode::GPU => unsafe { solver_advance_rk_gpu(self.0, a, dt) }
        }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        match self.1 {
            ExecutionMode::CPU => unsafe { solver_del_cpu(self.0) }
            ExecutionMode::OMP => unsafe { solver_del_omp(self.0) }
            ExecutionMode::GPU => unsafe { solver_del_gpu(self.0) }
        }
    }
}

extern "C" {
    pub(crate) fn solver_new_cpu(mesh: Mesh) -> *mut c_void;
    pub(crate) fn solver_new_omp(mesh: Mesh) -> *mut c_void;
    pub(crate) fn solver_new_gpu(mesh: Mesh) -> *mut c_void;
    pub(crate) fn solver_del_cpu(solver: *mut c_void);
    pub(crate) fn solver_del_omp(solver: *mut c_void);
    pub(crate) fn solver_del_gpu(solver: *mut c_void);
    pub(crate) fn solver_get_mesh_cpu(solver: *mut c_void) -> Mesh;
    pub(crate) fn solver_get_mesh_omp(solver: *mut c_void) -> Mesh;
    pub(crate) fn solver_get_mesh_gpu(solver: *mut c_void) -> Mesh;
    pub(crate) fn solver_get_primitive_cpu(solver: *mut c_void, primitive: *mut f64);
    pub(crate) fn solver_get_primitive_omp(solver: *mut c_void, primitive: *mut f64);
    pub(crate) fn solver_get_primitive_gpu(solver: *mut c_void, primitive: *mut f64);
    pub(crate) fn solver_set_primitive_cpu(solver: *mut c_void, primitive: *const f64);
    pub(crate) fn solver_set_primitive_omp(solver: *mut c_void, primitive: *const f64);
    pub(crate) fn solver_set_primitive_gpu(solver: *mut c_void, primitive: *const f64);
    pub(crate) fn solver_new_timestep_cpu(solver: *mut c_void);
    pub(crate) fn solver_new_timestep_omp(solver: *mut c_void);
    pub(crate) fn solver_new_timestep_gpu(solver: *mut c_void);
    pub(crate) fn solver_advance_rk_cpu(solver: *mut c_void, a: f64, dt: f64);
    pub(crate) fn solver_advance_rk_omp(solver: *mut c_void, a: f64, dt: f64);
    pub(crate) fn solver_advance_rk_gpu(solver: *mut c_void, a: f64, dt: f64);
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
