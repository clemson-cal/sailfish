use crate::solver::{BufferZone, EquationOfState, Mesh, PointMass};
use crate::error;
use kepler_two_body::{OrbitalElements, OrbitalState};
use error::Error::*;

pub trait Setup {
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]);
    fn masses(&self, time: f64) -> Vec<PointMass>;
    fn equation_of_state(&self) -> EquationOfState;
    fn buffer_zone(&self) -> BufferZone;
    fn max_signal_speed(&self) -> Option<f64>;
    fn viscosity(&self) -> Option<f64>;
    fn mesh(&self, resolution: u32) -> Mesh;
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

impl std::str::FromStr for Explosion {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self {})
        } else {
            Err(InvalidSetup("explosion problem does not take any parameters".to_string()))
        }
    }
}

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
    fn masses(&self, _time: f64) -> Vec<PointMass> {
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
    fn viscosity(&self) -> Option<f64> {
        None
    }
    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::centered_square(1.0, resolution)
    }
}

pub struct Binary {
    pub domain_radius: f64,
    pub nu: f64,
    pub sink_radius: f64,
    pub sink_rate: f64,
}

impl std::str::FromStr for Binary {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {

        let form = kind_config::Form::new()
            .item("domain_radius", 12.0, "half-size of the simulation domain [a]")
            .item("nu", 1e-3, "kinematic viscosity coefficient [Omega a^2]")
            .item("sink_radius", 0.05, "sink kernel radius [a]")
            .item("sink_rate", 10.0, "rate of mass subtraction in the sink [Omega]")
            .merge_string_args(parameters.split(':').filter(|s| !s.is_empty()))
            .map_err(|e| InvalidSetup(format!("{}", e)))?;

        for (key, val) in &form {
            println!("{:.<20} {:<10} {}", key, val.value, val.about)
        }

        Ok(Self {
            domain_radius: form.get("domain_radius").into(),
            nu: form.get("nu").into(),
            sink_radius: form.get("sink_radius").into(),
            sink_rate: form.get("sink_rate").into(),
        })
    }
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
    fn masses(&self, time: f64) -> Vec<PointMass> {
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
    fn viscosity(&self) -> Option<f64> {
        None
    }
    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::centered_square(8.0, resolution)
    }
}
