use crate::{
    node_2d, BoundaryCondition, Coordinates, Device, EquationOfState, ExecutionMode, IndexSpace,
    Mesh, Patch, PointMassList, StructuredMesh,
};

use gridiron::adjacency_list::AdjacencyList;
use gridiron::automaton::Automaton;
use gridiron::rect_map::Rectangle;

use std::sync::Arc;

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
    fn advance_rk(&mut self, setup: &dyn Setup, time: f64, a: f64, dt: f64);

    /// Primitive variable array in a solver using first, second, or third-order
    /// Runge-Kutta time stepping.
    fn advance(&mut self, setup: &dyn Setup, rk_order: u32, time: f64, dt: f64) {
        self.primitive_to_conserved();
        match rk_order {
            1 => {
                self.advance_rk(setup, time, 0.0, dt);
            }
            2 => {
                self.advance_rk(setup, time + 0.0 * dt, 0.0, dt);
                self.advance_rk(setup, time + 1.0 * dt, 0.5, dt);
            }
            3 => {
                // t1 = a1 * tn + (1 - a1) * (tn + dt) =     tn +     (      dt) = tn +     dt [a1 = 0]
                // t2 = a2 * tn + (1 - a2) * (t1 + dt) = 3/4 tn + 1/4 (tn + 2dt) = tn + 1/2 dt [a2 = 3/4]
                self.advance_rk(setup, time + 0.0 * dt, 0. / 1., dt);
                self.advance_rk(setup, time + 1.0 * dt, 3. / 4., dt);
                self.advance_rk(setup, time + 0.5 * dt, 1. / 3., dt);
            }
            _ => {
                panic!("invalid RK order")
            }
        }
    }
}

/// A trait to build patch-based solver instances.
///
/// With domain-decomposed grids, the driver needs to construct one solver
/// instance for each grid patch. So, rather than supplying the driver with
/// the solver instance(s), the top-level driver invocation provides a solver
/// builder, which the driver then uses to build as many solvers as there are
/// patches.
pub trait PatchBasedBuild {
    type Solver: PatchBasedSolve;

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        time: f64,
        primitive: Patch,
        global_structured_mesh: StructuredMesh,
        edge_list: &AdjacencyList<Rectangle<i64>>,
        rk_order: usize,
        mode: ExecutionMode,
        device: Option<Device>,
        setup: Arc<dyn Setup>,
    ) -> Self::Solver;
}

/// A trait for 2D solvers which operate on grid patches.
///
/// These solvers implement message passing and task-based parallelism via the
/// `Automaton` trait.
pub trait PatchBasedSolve:
    Automaton<Key = Rectangle<i64>, Value = Self, Message = Patch> + Send + Sync
{
    /// Returns the primitive variable array for this solver.
    ///
    /// The data is row-major with contiguous primitive variable components.
    /// The array does not include guard zones.
    fn primitive(&self) -> Patch;

    /// Sets the time step size to be used in subsequent advance stages.
    fn set_timestep(&mut self, dt: f64);

    /// Returns the largest wavespeed among the zones in the solver's current
    /// primitive array.
    ///
    /// This function should be as performant as possible, although if the
    /// reduction required to obtain the maximum wavespeed is slow, the effect
    /// might be mitigated by living on the edge and re-computing the timestep
    /// less frequently than every time step.
    fn max_wavespeed(&self) -> f64;

    /// Returns the GPU device this patch should be computed on, or `None` if
    /// the execution should be on the CPU.
    fn device(&self) -> Option<Device>;

    /// Returns a short sequence of floating-point numbers summarizing the
    /// solver state.
    ///
    /// This function is probably not called every iteration, so it's not
    /// expected to be as performant as the advance function. The driver will
    /// request the reductions from each grid patch at a user-specified time
    /// interval.
    ///
    /// The data should be extrinsic quantities, integrated rather than
    /// averaged over this grid patch, because the driver will sum the results
    /// over all grid patches to create a global reduction. Also for that
    /// reason, all grid patches must return a vector of the same length. The
    /// driver will append the user time to the start of the vector before
    /// recording a time series entry.
    fn reductions(&self) -> Vec<f64> {
        vec![]
    }
}

/// A trait describing a simulation model setup.
///
/// This trait is used by the driver to define initial and boundary
/// conditions, select a hydrodynamics solver and parameters, and describe
/// physics conditions such as gravity and thermodynamics. Basic setups only
/// need to implement a subset of the possible methods; most of the methods
/// below have stub default implementations.
pub trait Setup: Send + Sync {
    /// Invoked by the solver to determine an equation of state (EOS), if that
    /// solver supports different types.
    ///
    /// Not all solvers support all EOS's, and the solver is not expected to
    /// produce an error if an incompatible EOS is returned from this method.
    fn equation_of_state(&self) -> EquationOfState;

    /// Invoked by the driver to build a mesh on which to run this problem.
    ///
    /// The mesh type must be compatible with the requested solver.
    ///
    /// The resolution parameter will be collected from the command line or
    /// restart file and provided to this method. The problem setup is then
    /// free to interpret the resolution parameter as it sees fit to generate
    /// a `Mesh` instance. For example, the resolution parameter may be
    /// interpreted by the setup as "number of zones per side in a 2D square
    /// grid", "number of zones per decade" in a 1D spherical polar grid, or
    /// "number of polar zones" in a 2D spherical polar grid.
    fn mesh(&self, resolution: u32) -> Mesh;

    /// Invoked by the solver to determine the coordinate system.
    ///
    /// Not all solvers support all coordinate systemsm and the solver is not
    /// expected to produce an error if an imcompatible coordinate system is
    /// returned from this method.
    fn coordinate_system(&self) -> Coordinates;

    /// Invoked by the driver to determine of primitive variable fields which
    /// are to be evolved.
    fn num_primitives(&self) -> usize;

    /// Required method, invoked by the driver, to identify which physics
    /// solver the setup wants to run on.
    fn solver_name(&self) -> String;

    /// Required method, invoked by the driver and possibly the solver, to
    /// specify initial and boundary conditions.
    ///
    /// Note: This method might be changed to `primitive_at` or something
    /// similar, and be given a time argument to facilitate a time-dependent
    /// boundary condition.
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]);

    /// The time the simulation should start counting from.
    ///
    /// Usually this will be `t=0`, however it sometimes makes sense to start
    /// from `t=1` or something else, especially with explosion problems or
    /// when log-spaced checkpoint outputs are desired.
    fn initial_time(&self) -> f64 {
        0.0
    }

    /// The time when the simulation should terminate. `None` means to evolve
    /// indefinitely.
    fn end_time(&self) -> Option<f64> {
        None
    }

    /// Invoked by the driver to convert from "simulation time" to "user
    /// time".
    ///
    /// Simulation time is the time used by the physics solvers. Messages
    /// written to `stdout` by the driver report time `t` and timestep size
    /// `dt` in user time. The simulation start time, end time, as well as
    /// task intervals, are provided from this in user time.
    fn unit_time(&self) -> f64 {
        1.0
    }

    /// Invoked by the driver to give the problem setup an opportunity to
    /// print its configuration to stdout.
    fn print_parameters(&self) {}

    /// This method should be implemented by setups which accept runtime model
    /// parameters. All setups implement `FromStr` (see the [`crate::setups`]
    /// module), and that method should be able to restore this setup from the
    /// string returned by this method.
    fn model_parameter_string(&self) -> String {
        String::new()
    }

    /// Invoked by solver modules which support a gravitational field sourced
    /// by point-like test masses.
    ///
    /// The time argument is in simulation time, not user time (see
    /// [`Setup::unit_time`]).
    ///
    /// This method is also called in the [`crate::state`] module to enable
    /// writing the point mass locations to checkpoint files for diagnostic
    /// purposes.
    fn masses(&self, _time: f64) -> PointMassList {
        PointMassList::default()
    }

    /// Invoked by solver modules which support a wave-damping zone.
    fn boundary_condition(&self) -> BoundaryCondition {
        BoundaryCondition::Default
    }

    /// Invoked by solver modules which support viscous stresses.
    fn viscosity(&self) -> Option<f64> {
        None
    }

    /// Invoked by solver modules which support thermodynamic cooling
    /// prescriptions.
    fn cooling_coefficient(&self) -> Option<f64> {
        None
    }

    fn mach_ceiling(&self) -> Option<f64> {
        None
    }

    fn density_floor(&self) -> Option<f64> {
        None
    }

    fn pressure_floor(&self) -> Option<f64> {
        None
    }

    fn velocity_ceiling(&self) -> Option<f64> {
        None
    }

    fn constant_softening(&self) -> Option<bool> {
        None
    }

    fn dg_cell(&self) -> Option<node_2d::Cell> {
        None
    }

    fn initial_primitive_vec(&self, mesh: &Mesh) -> Vec<f64> {
        match mesh {
            Mesh::Structured(_) => {
                panic!("Setup::initial_primitive_vec is only compatible with Mesh::FacePositions1D")
            }
            Mesh::FacePositions1D(faces) => {
                let nq = self.num_primitives();
                let mut primitive = vec![0.0; (faces.len() - 1) * nq];
                for i in 0..faces.len() - 1 {
                    let x = 0.5 * (faces[i] + faces[i + 1]);
                    self.initial_primitive(x, 0.0, &mut primitive[nq * i..nq * i + nq]);
                }
                primitive
            }
        }
    }

    /// Provided method to generate grid patches of primitive data from the
    /// initial model.
    fn initial_primitive_patch(&self, space: &IndexSpace, mesh: &Mesh) -> Patch {
        if let Some(cell) = self.dg_cell() {
            self.initial_primitive_patch_dg(space, mesh, &cell)
        } else {
            self.initial_primitive_patch_conventional(space, mesh)
        }
    }

    fn initial_primitive_patch_conventional(&self, space: &IndexSpace, mesh: &Mesh) -> Patch {
        match mesh {
            Mesh::Structured(mesh) => {
                Patch::from_slice_function(space, self.num_primitives(), |(i, j), prim| {
                    let [x, y] = mesh.cell_coordinates(i, j);
                    self.initial_primitive(x, y, prim)
                })
            }
            _ => unimplemented!(),
        }
    }

    /// Provided method to generate grid patches of primitive data from the
    /// initial model, with the primitive variables sampled at sub-cell
    /// quadrature points as specified by the `cell` variable.
    fn initial_primitive_patch_dg(&self, space: &IndexSpace, mesh: &Mesh, cell: &node_2d::Cell) -> Patch {
        match mesh {
            Mesh::Structured(mesh) => {
                let ni = cell.quadrature_points().count();
                let nq = self.num_primitives();

                Patch::from_slice_function(space, ni * nq, |(i, j), prim| {
                    for (n, [a, b]) in cell.quadrature_points().enumerate() {
                        let [x, y] = mesh.subcell_coordinates(i, j, a, b);
                        self.initial_primitive(x, y, &mut prim[n * nq..(n + 1) * nq])
                    }
                })
            }
            _ => unimplemented!(),
        }
    }
}
