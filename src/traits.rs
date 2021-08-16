use crate::{
    BufferZone, Coordinates, Device, EquationOfState, ExecutionMode, IndexSpace, Mesh, Patch,
    PointMassList, StructuredMesh,
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

pub trait PatchBasedBuild {
    type Solver: PatchBasedSolve;

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

pub trait PatchBasedSolve:
    Automaton<Key = Rectangle<i64>, Value = Self, Message = Patch> + Send + Sync
{
    /// Returns the primitive variable array for this solver. The data is
    /// row-major with contiguous primitive variable components. The array
    /// includes guard zones.
    fn primitive(&self) -> Patch;

    /// Sets the time step size to be used in subsequent advance stages.
    fn set_timestep(&mut self, dt: f64);

    /// Returns the largest wavespeed among the zones in the solver's current
    /// primitive array.
    fn max_wavespeed(&self) -> f64;

    /// Returns the GPU device this patch should be computed on, or `None` if
    /// the execution should be on the CPU.
    fn device(&self) -> Option<Device>;
}

pub trait Setup: Send + Sync {
    fn print_parameters(&self) {}
    fn solver_name(&self) -> String;
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]);
    fn initial_time(&self) -> f64 {
        0.0
    }
    fn end_time(&self) -> Option<f64> {
        None
    }
    fn unit_time(&self) -> f64 {
        1.0
    }
    fn masses(&self, _time: f64) -> PointMassList {
        PointMassList::default()
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::NoBuffer
    }
    fn viscosity(&self) -> Option<f64> {
        None
    }
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
    fn equation_of_state(&self) -> EquationOfState;
    fn mesh(&self, resolution: u32) -> Mesh;
    fn coordinate_system(&self) -> Coordinates;
    fn num_primitives(&self) -> usize;
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
    fn initial_primitive_patch(&self, space: &IndexSpace, mesh: &Mesh) -> Patch {
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
}
