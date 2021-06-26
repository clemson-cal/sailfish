use crate::solver::{BufferZone, EquationOfState, Mesh, PointMass};
use kepler_two_body::{OrbitalElements, OrbitalState};

pub trait Setup: Sized {
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]);
    fn particles(&self, time: f64) -> Vec<PointMass>;
    fn equation_of_state(&self) -> EquationOfState;
    fn buffer_zone(&self) -> BufferZone;
    fn max_signal_speed(&self) -> Option<f64>;

    fn initial_primitive_vec(&self, mesh: &Mesh) -> Vec<f64> {
        let mut primitive = vec![0.0; ((mesh.ni + 4) * (mesh.nj + 4) * 3) as usize];
        let si = 3 * (mesh.nj + 4);
        let sj = 3;
        for i in -2..mesh.ni + 2 {
            for j in -2..mesh.nj + 2 {
                let n = ((i + 2) * si + (j + 2) * sj) as usize;
                let [x, y] = mesh.cell_coordinates(i, j);
                self.initial_primitive(x, y, &mut primitive[n..n + 3])
            }
        }
        primitive
    }
}

pub struct Explosion {}

impl Setup for Explosion {
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]) {
        if (x * x + y * y).sqrt() < 0.25 {
            primitive[0] = 1.0;
        } else {
            primitive[0] = 0.1;
        }
        primitive[1] = 0.0;
        primitive[2] = 0.0;
    }
    fn particles(&self, _time: f64) -> Vec<PointMass> {
        vec![]
    }
    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::Isothermal { sound_speed_squared: 1.0 }
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::None
    }
    fn max_signal_speed(&self) -> Option<f64> {
        Some(1.0)
    }
}

pub struct Binary {
    sink_radius: f64,
    sink_rate: f64,
}

impl Setup for Binary {
    #[allow(clippy::many_single_char_names)]
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]) {
        let r = (x * x + y * y).sqrt();
        let rs = (x * x + y * y + self.sink_radius.powf(2.0)).sqrt();
        let phi_hat_x = -y / r;
        let phi_hat_y = x / r;
        let d = 1.0;
        let u = phi_hat_x / rs.sqrt();
        let v = phi_hat_y / rs.sqrt();
        primitive[0] = d;
        primitive[1] = u;
        primitive[2] = v;
    }
    fn particles(&self, time: f64) -> Vec<PointMass> {
        let a: f64 = 1.0;
        let m: f64 = 1.0;
        let q: f64 = 1.0;
        let e: f64 = 0.0;
        let binary = OrbitalElements(a, m, q, e);
        let OrbitalState(mass0, mass1) = binary.orbital_state_from_time(time);
        let mass0 = PointMass {
            x: mass0.position_x(),
            y: mass0.position_y(),
            mass: mass0.mass(),
            rate: self.sink_rate,
            radius: self.sink_radius,
        };
        let mass1 = PointMass {
            x: mass1.position_x(),
            y: mass1.position_y(),
            mass: mass1.mass(),
            rate: self.sink_rate,
            radius: self.sink_radius,
        };
        vec![mass0, mass1]
    }
    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::LocallyIsothermal { mach_number_squared: 10.0f64.powi(2) }
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::None
    }
    fn max_signal_speed(&self) -> Option<f64> {
        Some(1.0 / self.sink_radius.sqrt())
    }
}
