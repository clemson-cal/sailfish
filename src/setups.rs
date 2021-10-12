//! Master list of the problem setups.

use crate::error::{self, Error::*};
use crate::lookup_table::LookupTable;
use crate::mesh::Mesh;
use crate::{
    BoundaryCondition, Coordinates, EquationOfState, PointMass, PointMassList, Setup, SinkModel,
    StructuredMesh,
};
use std::f64::consts::PI;

use kepler_two_body::{OrbitalElements, OrbitalState};
use std::fmt::Write;
use std::str::FromStr;
use std::sync::Arc;

macro_rules! setup_builder {
    ($setup:ident) => {
        Box::new(|p| Ok(Arc::new($setup::from_str(p)?)))
    };
}

type SetupFunction = Box<dyn Fn(&str) -> Result<Arc<dyn Setup>, error::Error>>;

fn setups() -> Vec<(&'static str, SetupFunction)> {
    vec![
        ("binary", setup_builder!(Binary)),
        ("binary-therm", setup_builder!(BinaryWithThermodynamics)),
        ("envelope-shock", setup_builder!(EnvelopeShock)),
        ("explosion", setup_builder!(Explosion)),
        ("fast-shell", setup_builder!(FastShell)),
        ("pulse-collision", setup_builder!(PulseCollision)),
        ("sedov", setup_builder!(Sedov)),
        ("shocktube", setup_builder!(Shocktube)),
        ("wind", setup_builder!(Wind)),
    ]
}

/// Generates an error message of type `PrintUserInformation` listing known
/// problem setups.
pub fn possible_setups_info() -> error::Error {
    let mut message = String::new();
    writeln!(message, "specify setup:").unwrap();
    for (setup_name, _) in setups() {
        writeln!(message, "    {}", setup_name).unwrap();
    }
    PrintUserInformation(message)
}

/// Tries to construct a dynamic setup from a string key and model parameter
/// string.
///
/// The result is put under `Arc` so it can be attached to solver instances
/// and shared safely between threads. If no setup matches the given name, a
/// `PrintUserInformation` error is returned listing the available setups. If
/// a setup is found, but has an invalid configuration, the `InvalidSetup`
/// error is returned here.
pub fn make_setup(setup_name: &str, parameters: &str) -> Result<Arc<dyn Setup>, error::Error> {
    setups()
        .into_iter()
        .find(|&(n, _)| n == setup_name)
        .map(|(_, f)| f(parameters))
        .ok_or_else(possible_setups_info)?
}

/// Classic 1D shocktube problem for the energy-conserving Euler equation
pub struct Shocktube;

impl FromStr for Shocktube {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self)
        } else {
            Err(InvalidSetup("setup does not take any parameters".into()))
        }
    }
}

impl Setup for Shocktube {
    fn num_primitives(&self) -> usize {
        3
    }

    fn solver_name(&self) -> String {
        "euler1d".to_owned()
    }

    fn initial_primitive(&self, x: f64, _y: f64, primitive: &mut [f64]) {
        if x < 0.5 {
            primitive[0] = 1.0;
            primitive[2] = 1.0;
        } else {
            primitive[0] = 0.1;
            primitive[2] = 0.125;
        }
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 5.0 / 3.0,
        }
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        let dx = 1.0 / resolution as f64;
        let faces = (0..resolution + 1).map(|i| i as f64 * dx).collect();
        Mesh::FacePositions1D(faces)
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::Cartesian
    }

    fn end_time(&self) -> Option<f64> {
        Some(0.15)
    }
}

/// A cylindrical explosion in 2D planar geometry; isothermal hydro.
///
/// This problem is useful for testing bare-bones setups with minimal physics.
/// A circular region of high density and pressure is initiated at the center
/// of a square domain. The gas has isothermal equation of state with global
/// sound speed `cs=1`.
pub struct Explosion;

impl FromStr for Explosion {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self)
        } else {
            Err(InvalidSetup("setup does not take any parameters".into()))
        }
    }
}

impl Setup for Explosion {
    fn num_primitives(&self) -> usize {
        3
    }
    fn solver_name(&self) -> String {
        "iso2d".to_owned()
    }
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]) {
        if (x * x + y * y).sqrt() < 0.25 {
            primitive[0] = 1.0;
        } else {
            primitive[0] = 0.1;
        }
        primitive[1] = 0.0;
        primitive[2] = 0.0;
    }
    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::Isothermal {
            sound_speed_squared: 1.0,
        }
    }
    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::Structured(StructuredMesh::centered_square(1.0, resolution))
    }
    fn coordinate_system(&self) -> Coordinates {
        Coordinates::Cartesian
    }
    fn end_time(&self) -> Option<f64> {
        Some(0.2)
    }
}

pub struct Binary {
    pub domain_radius: f64,
    pub nu: f64,
    pub mach_number: f64,
    pub sink_radius1: f64,
    pub sink_radius2: f64,
    pub sink_rate1: f64,
    pub sink_rate2: f64,
    pub sink_model: SinkModel,
    form: kind_config::Form,
}

impl FromStr for Binary {
    type Err = error::Error;

    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        #[rustfmt::skip]
        let form = kind_config::Form::new()
            .item("domain_radius", 12.0, "half-size of the simulation domain (a)")
            .item("nu",            1e-3, "kinematic viscosity coefficient (Omega a^2)")
            .item("mach_number",   10.0, "mach number for locally isothermal EOS")
            .item("sink_radius", "0.05", "sink kernel radii (a)")
            .item("sink_model",    "af", "sink prescription: [none|af|tf|ff]")
            .item("sink_rate",   "10.0", "rate(s) of mass subtraction in the sink (Omega)")
            .item("q",              1.0, "system mass ratio: [0-1]")
            .item("e",              0.0, "orbital eccentricity: [0-1]")
            .merge_string_args_allowing_duplicates(parameters.split(':').filter(|s| !s.is_empty()))
            .map_err(|e| InvalidSetup(format!("{}", e)))?;

        let (sradius1, sradius2) = crate::parse::parse_pair(form.get("sink_radius").into(), ',')
            .map_err(ParseFloatError)?;

        let (srate1, srate2) =
            crate::parse::parse_pair(form.get("sink_rate").into(), ',').map_err(ParseFloatError)?;

        Ok(Self {
            domain_radius: form.get("domain_radius").into(),
            nu: form.get("nu").into(),
            mach_number: form.get("mach_number").into(),
            sink_radius1: sradius1.unwrap(),
            sink_radius2: sradius2.or(sradius1).unwrap(),
            sink_rate1: srate1.unwrap(),
            sink_rate2: srate2.or(srate1).unwrap(),
            sink_model: SinkModel::from_str(form.get("sink_model").into())?,
            form,
        })
    }
}

impl Setup for Binary {
    fn num_primitives(&self) -> usize {
        3
    }

    fn print_parameters(&self) {
        for key in self.form.sorted_keys() {
            println!(
                "{:.<20} {:<10} {}",
                key,
                self.form.get(&key),
                self.form.about(&key)
            );
        }
        println!(
            "sink radii are [{}, {}]",
            self.sink_radius1, self.sink_radius2
        );
        println!("sink rates are [{}, {}]", self.sink_rate1, self.sink_rate2);
    }

    fn model_parameter_string(&self) -> String {
        self.form
            .iter()
            .map(|(a, b)| format!("{}={}", a, b))
            .collect::<Vec<_>>()
            .join(":")
    }

    fn solver_name(&self) -> String {
        "iso2d".to_owned()
    }

    fn unit_time(&self) -> f64 {
        2.0 * std::f64::consts::PI
    }

    #[allow(clippy::many_single_char_names)]
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]) {
        let r = (x * x + y * y).sqrt();
        let rs = (x * x + y * y + self.sink_radius1.powf(2.0)).sqrt(); //use primary sink radius for both masses
        let phi_hat_x = -y / r.max(1e-12);
        let phi_hat_y = x / r.max(1e-12);
        let d = 1.0;
        let u = phi_hat_x / rs.sqrt();
        let v = phi_hat_y / rs.sqrt();
        primitive[0] = d;
        primitive[1] = u;
        primitive[2] = v;
    }

    fn masses(&self, time: f64) -> PointMassList {
        let a: f64 = 1.0;
        let m: f64 = 1.0;
        let q: f64 = self.form.get("q").into();
        let e: f64 = self.form.get("e").into();
        let binary = OrbitalElements(a, m, q, e);
        let OrbitalState(mass1, mass2) = binary.orbital_state_from_time(time);
        let mass1 = PointMass {
            x: mass1.position_x(),
            y: mass1.position_y(),
            vx: mass1.velocity_x(),
            vy: mass1.velocity_y(),
            mass: mass1.mass(),
            rate: self.sink_rate1,
            radius: self.sink_radius1,
            model: self.sink_model,
        };
        let mass2 = PointMass {
            x: mass2.position_x(),
            y: mass2.position_y(),
            vx: mass2.velocity_x(),
            vy: mass2.velocity_y(),
            mass: mass2.mass(),
            rate: self.sink_rate2,
            radius: self.sink_radius2,
            model: self.sink_model,
        };
        PointMassList::from_slice(&[mass1, mass2])
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::LocallyIsothermal {
            mach_number_squared: self.mach_number.powi(2),
        }
    }

    fn boundary_condition(&self) -> BoundaryCondition {
        BoundaryCondition::Default
    }

    fn viscosity(&self) -> Option<f64> {
        Some(self.nu)
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::Structured(StructuredMesh::centered_square(
            self.domain_radius,
            resolution,
        ))
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::Cartesian
    }
}

pub struct BinaryWithThermodynamics {
    pub domain_radius: f64,
    pub alpha: f64,
    pub sink_radius1: f64,
    pub sink_radius2: f64,
    pub sink_rate1: f64,
    pub sink_rate2: f64,
    pub sink_model: SinkModel,
    pub gamma_law_index: f64,
    pub cooling_coefficient: f64,
    pub pressure_floor: f64,
    pub density_floor: f64,
    pub velocity_ceiling: f64,
    pub initial_density: f64,
    pub initial_pressure: f64,
    pub mach_ceiling: f64,
    pub test_model: bool,
    pub one_body: bool,
    pub constant_softening: bool,
    form: kind_config::Form,
}

impl FromStr for BinaryWithThermodynamics {
    type Err = error::Error;

    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        #[rustfmt::skip]
        let form = kind_config::Form::new()
            .item("domain_radius",      12.0, "half-size of the simulation domain (a)")
            .item("alpha",               0.1, "alpha-viscosity coefficient (dimensionless)")
            .item("sink_radius",      "0.05", "sink kernel radii (a)")
            .item("sink_model",         "af", "sink prescription: [none|af|tf|ff]")
            .item("sink_rate",        "10.0", "rate(s) of mass subtraction in the sink (Omega)")
            .item("q",                   1.0, "system mass ratio: [0-1]")
            .item("e",                   0.0, "orbital eccentricity: [0-1]")
            .item("gamma_law_index",     1.666666666666666, "adiabatic index")
            .item("cooling_coefficient", 0.0, "strength of T^4 cooling")
            .item("pressure_floor",      0.0, "pressure floor")
            .item("density_floor",       0.0, "density floor")
            .item("velocity_ceiling",   1e16, "component-wise ceiling on the velocity (a * Omega)")
            .item("initial_density",     1.0, "initial surface density at r=a")
            .item("initial_pressure",   1e-2, "initial surface  pressure at r=a")
            .item("mach_ceiling",        1e5, "cooling respects the mach ceiling")
            .item("test_model",        false, "use test model")
            .item("one_body",          false, "use one point mass")
            .item("constant_softening",false, "use constant gravitational softening = sink_radius")
            .merge_string_args_allowing_duplicates(parameters.split(':').filter(|s| !s.is_empty()))
            .map_err(|e| InvalidSetup(format!("{}", e)))?;

        let (sradius1, sradius2) = crate::parse::parse_pair(form.get("sink_radius").into(), ',')
            .map_err(ParseFloatError)?;

        let (srate1, srate2) =
            crate::parse::parse_pair(form.get("sink_rate").into(), ',').map_err(ParseFloatError)?;

        Ok(Self {
            domain_radius: form.get("domain_radius").into(),
            alpha: form.get("alpha").into(),
            sink_radius1: sradius1.unwrap(),
            sink_radius2: sradius2.or(sradius1).unwrap(),
            sink_rate1: srate1.unwrap(),
            sink_rate2: srate2.or(srate1).unwrap(),
            sink_model: SinkModel::from_str(form.get("sink_model").into())?,
            gamma_law_index: form.get("gamma_law_index").into(),
            cooling_coefficient: form.get("cooling_coefficient").into(),
            pressure_floor: form.get("pressure_floor").into(),
            density_floor: form.get("density_floor").into(),
            velocity_ceiling: form.get("velocity_ceiling").into(),
            initial_density: form.get("initial_density").into(),
            initial_pressure: form.get("initial_pressure").into(),
            mach_ceiling: form.get("mach_ceiling").into(),
            test_model: form.get("test_model").into(),
            one_body: form.get("one_body").into(),
            constant_softening: form.get("constant_softening").into(),
            form,
        })
    }
}

impl BinaryWithThermodynamics {
    fn density_scaling(&self, r: f64) -> f64 {
        r.powf(-3.0 / 5.0) // Eq. (A2) from Goodman (2003)
    }

    fn pressure_scaling(&self, r: f64) -> f64 {
        r.powf(-3.0 / 2.0) // Derived from Goodman (2003)
    }
}

impl Setup for BinaryWithThermodynamics {
    fn solver_name(&self) -> String {
        "euler2d".to_owned()
    }

    fn num_primitives(&self) -> usize {
        4
    }

    fn model_parameter_string(&self) -> String {
        self.form
            .iter()
            .map(|(a, b)| format!("{}={}", a, b))
            .collect::<Vec<_>>()
            .join(":")
    }

    fn print_parameters(&self) {
        for key in self.form.sorted_keys() {
            println!(
                "{:.<20} {:<10} {}",
                key,
                self.form.get(&key),
                self.form.about(&key)
            );
        }
    }

    fn unit_time(&self) -> f64 {
        2.0 * std::f64::consts::PI
    }

    #[allow(clippy::many_single_char_names)]
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]) {
        if !self.test_model {
            let r = (x * x + y * y).sqrt();
            let rs = (x * x + y * y + self.sink_radius1.powf(2.0)).sqrt();
            let phi_hat_x = -y / r.max(1e-12);
            let phi_hat_y = x / r.max(1e-12);
            let d = self.initial_density
                * self.density_scaling(rs)
                * (0.0001 + 0.9999 * f64::exp(-(2.0 / rs).powi(30)));
            let p = self.initial_pressure
                * self.pressure_scaling(rs)
                * (0.0001 + 0.9999 * f64::exp(-(2.0 / rs).powi(30)));
            let u = phi_hat_x / rs.sqrt();
            let v = phi_hat_y / rs.sqrt();
            primitive[0] = d;
            primitive[1] = u;
            primitive[2] = v;
            primitive[3] = p;
        } else {
            let r = (x * x + y * y).sqrt();
            let phi = f64::atan2(x, y);
            let r1 = ((x - 1.0).powi(2) + (y - 1.0).powi(2)).sqrt();
            let r2 = ((x + 1.0).powi(2) + (y + 1.0).powi(2)).sqrt();
            let d = 1.0 + f64::exp(-r1.powi(2));
            let p = 1.0 + f64::exp(-r2.powi(2));
            let vp = r.powf(-0.5) * f64::exp(-5.0 / r - r.powi(2) / 3.0);
            let vr = f64::sin(phi - 3.14159 / 4.0) * f64::exp(-5.0 / r - r.powi(2) / 3.0);
            let u = vp * (-y / r) + vr * (x / r);
            let v = vp * (x / r) + vr * (y / r);
            primitive[0] = d;
            primitive[1] = u;
            primitive[2] = v;
            primitive[3] = p;
        }
    }

    fn masses(&self, time: f64) -> PointMassList {
        if !self.one_body {
            let a: f64 = 1.0;
            let m: f64 = 1.0;
            let q: f64 = self.form.get("q").into();
            let e: f64 = self.form.get("e").into();
            let binary = OrbitalElements(a, m, q, e);
            let OrbitalState(mass1, mass2) = binary.orbital_state_from_time(time);
            let mass1 = PointMass {
                x: mass1.position_x(),
                y: mass1.position_y(),
                vx: mass1.velocity_x(),
                vy: mass1.velocity_y(),
                mass: mass1.mass(),
                rate: self.sink_rate1,
                radius: self.sink_radius1,
                model: self.sink_model,
            };
            let mass2 = PointMass {
                x: mass2.position_x(),
                y: mass2.position_y(),
                vx: mass2.velocity_x(),
                vy: mass2.velocity_y(),
                mass: mass2.mass(),
                rate: self.sink_rate2,
                radius: self.sink_radius2,
                model: self.sink_model,
            };
            PointMassList::from_slice(&[mass1, mass2])
        } else {
            let mass1 = PointMass {
                x: 0.0,
                y: 0.0,
                vx: 0.0,
                vy: 0.0,
                mass: 1.0,
                rate: self.sink_rate1,
                radius: self.sink_radius1,
                model: self.sink_model,
            };
            PointMassList::from_slice(&[mass1])
        }
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: self.gamma_law_index,
        }
    }

    fn boundary_condition(&self) -> BoundaryCondition {
        if !self.test_model {
            let onset_radius = self.domain_radius - 0.1;
            BoundaryCondition::KeplerianBuffer {
                surface_density: self.initial_density * self.density_scaling(onset_radius),
                surface_pressure: self.initial_pressure * self.pressure_scaling(onset_radius),
                central_mass: 1.0,
                driving_rate: 1000.0,
                outer_radius: self.domain_radius,
                onset_width: 0.1,
            }
        } else {
            BoundaryCondition::KeplerianBuffer {
                // don't change this
                surface_density: 1.0,
                surface_pressure: 1.0,
                central_mass: 0.0,
                driving_rate: 10.0,
                outer_radius: self.domain_radius,
                onset_width: 0.1,
            }
        }
    }

    fn viscosity(&self) -> Option<f64> {
        Some(self.alpha)
    }

    fn velocity_ceiling(&self) -> Option<f64> {
        Some(self.velocity_ceiling)
    }

    fn cooling_coefficient(&self) -> Option<f64> {
        Some(self.cooling_coefficient)
    }

    fn mach_ceiling(&self) -> Option<f64> {
        Some(self.mach_ceiling)
    }

    fn density_floor(&self) -> Option<f64> {
        Some(self.density_floor)
    }

    fn pressure_floor(&self) -> Option<f64> {
        Some(self.pressure_floor)
    }

    fn constant_softening(&self) -> Option<bool> {
        Some(self.constant_softening)
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::Structured(StructuredMesh::centered_square(
            self.domain_radius,
            resolution,
        ))
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::Cartesian
    }
}

/// Sedov-Taylor explosion setup, with tabulated initial condition.
///
/// This problem uses an ASCII table for the initial data. The table must
/// contain rows of data with columns `(r, rho, vr, pre)`. The radial
/// coordinate is that of the cell center. Faces are constructed at the
/// midpoints between the cell radii.
pub struct Sedov {
    faces: Vec<f64>,
    table: LookupTable<4>,
    filename: String,
}

impl FromStr for Sedov {
    type Err = error::Error;

    fn from_str(filename: &str) -> Result<Self, Self::Err> {
        use std::iter::once;

        let table = if filename.is_empty() {
            Err(InvalidSetup("usage -- sedov:input.dat".to_owned()))
        } else {
            LookupTable::<4>::from_ascii_file(filename).map_err(|e| InvalidSetup(format!("{}", e)))
        }?;

        if table.len() < 2 {
            return Err(InvalidSetup("table must have at least 2 rows".to_owned()));
        }
        let cell_centers: Vec<f64> = table.rows().iter().map(|row| row[0]).collect();
        let n = cell_centers.len();
        let dxl = cell_centers[1] - cell_centers[0];
        let dxr = cell_centers[n - 1] - cell_centers[n - 2];
        let faces = once(cell_centers[0] - 0.5 * dxl)
            .chain(cell_centers.windows(2).map(|w| 0.5 * (w[0] + w[1])))
            .chain(once(cell_centers[n - 1] + 0.5 * dxr))
            .collect();
        Ok(Self {
            faces,
            table,
            filename: filename.to_string(),
        })
    }
}

impl Setup for Sedov {
    fn num_primitives(&self) -> usize {
        3
    }

    fn solver_name(&self) -> String {
        "euler1d".to_owned()
    }

    fn model_parameter_string(&self) -> String {
        self.filename.clone()
    }

    fn initial_primitive(&self, x: f64, _y: f64, primitive: &mut [f64]) {
        let row = self.table.sample(x);
        primitive[0] = row[1];
        primitive[1] = row[2];
        primitive[2] = row[3];
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 5.0 / 3.0,
        }
    }

    fn boundary_condition(&self) -> BoundaryCondition {
        BoundaryCondition::Default
    }

    fn viscosity(&self) -> Option<f64> {
        None
    }

    fn mesh(&self, _resolution: u32) -> Mesh {
        // Note: resolution is ignored. Consider making it an Option, and
        // returning `Result` in case it's given for problems that specify the
        // resolution internally.
        Mesh::FacePositions1D(self.faces.clone())
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::SphericalPolar
    }

    fn initial_time(&self) -> f64 {
        1.0
    }
}

/// Collision of counter-propagating planar mass shells.
///
/// The fluid is non-relativistic. The setup does not have runtime model
/// parameters, it is hard-coded for a Mach number of 50, a domain extending
/// from x=-100, to x=100, and the pulses have a mass ratio of 100:1. They
/// move in the opposite direction but with equal momentum so the simulation
/// is in the center-of-momentum frame.
pub struct PulseCollision;

impl FromStr for PulseCollision {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self)
        } else {
            Err(InvalidSetup("setup does not take any parameters".into()))
        }
    }
}

impl Setup for PulseCollision {
    fn num_primitives(&self) -> usize {
        3
    }

    fn solver_name(&self) -> String {
        "euler1d".to_owned()
    }

    fn initial_primitive(&self, x: f64, _y: f64, primitive: &mut [f64]) {
        let xl: f64 = -2.0;
        let xr: f64 = 2.0;
        let dx: f64 = 0.05;

        let gaussian = |x: f64, x0: f64| f64::exp(-(x - x0).powi(2) / dx.powi(2));

        let step = |x: f64, x0: f64| {
            if (x - x0).abs() < dx * 10.0 {
                1.0
            } else {
                0.0
            }
        };

        let rho = 100.0 * gaussian(x, xl) + gaussian(x, xr) + 1e-7;
        let vel = 0.01 * step(x, xl) - step(x, xr);
        let mach = 50.0;
        let vel_sound = vel / mach;
        let p = 3.0 / 5.0 * vel_sound * vel_sound * rho + 1e-11;

        primitive[0] = rho;
        primitive[1] = vel;
        primitive[2] = p;
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 5.0 / 3.0,
        }
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        let x0 = -100.0;
        let x1 = 100.0;
        let dx = (x1 - x0) / resolution as f64;
        let faces = (0..resolution + 1).map(|i| x0 + i as f64 * dx).collect();
        Mesh::FacePositions1D(faces)
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::Cartesian
    }

    fn end_time(&self) -> Option<f64> {
        Some(1.00)
    }
}

/// Collision of a fast shell with a wind-like target medium in spherical
/// geometry.
///
/// The fluid is non-relativistic. The setup does not have runtime model
/// parameters yet.
pub struct FastShell;

impl FromStr for FastShell {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self)
        } else {
            Err(InvalidSetup("setup does not take any parameters".into()))
        }
    }
}

impl Setup for FastShell {
    fn num_primitives(&self) -> usize {
        3
    }

    fn solver_name(&self) -> String {
        "euler1d".to_owned()
    }

    fn initial_primitive(&self, r: f64, _y: f64, primitive: &mut [f64]) {
        // The deceleration radius is defined as the radius at which the shell
        // will have transferred its kinetic energy to the ambient medium. In
        // non-relativistic hydrodynamics, this is the radius the shell
        // expands to when it as swept up its own mass. The shell width is dr,
        // and its mass is roughly rho_1 * dr * 4 * pi * r^2. The deceleration
        // radius is r_dec = dr * rho_1 / rho_0. For the fiducial setup below,
        // r_dec is 100.

        let r_shell: f64 = 10.0;
        let dr = 1.0;
        let rho_0 = 0.01;
        let rho_1 = 1.0;
        let v_max = 1.0;

        let prof = |r: f64| {
            if r > r_shell {
                0.0
            } else {
                f64::exp((r - r_shell) / dr)
            }
        };

        let rho_ambient = rho_0 * (r / r_shell).powi(-2);
        let rho = rho_1 * prof(r) + rho_ambient;
        let vel = v_max * prof(r);
        let pre = 1e-3 * rho_ambient;

        primitive[0] = rho;
        primitive[1] = vel;
        primitive[2] = pre;
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 5.0 / 3.0,
        }
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::logarithmic_radial(1.0, 2.0, resolution)
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::SphericalPolar
    }

    fn end_time(&self) -> Option<f64> {
        Some(1.0)
    }
}

/// A spherical wind setup.
pub struct Wind;

impl FromStr for Wind {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self)
        } else {
            Err(InvalidSetup("setup does not take any parameters".into()))
        }
    }
}

impl Setup for Wind {
    fn num_primitives(&self) -> usize {
        3
    }

    fn solver_name(&self) -> String {
        "sr1d".to_owned()
    }

    fn initial_primitive(&self, r: f64, _y: f64, primitive: &mut [f64]) {
        let rho = r.powi(-2);
        let vel = 1.0;
        let pre = 1e-6 * rho;

        primitive[0] = rho;
        primitive[1] = vel;
        primitive[2] = pre;
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 4.0 / 3.0,
        }
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::logarithmic_radial(1.0, 2.0, resolution)
    }

    fn boundary_condition(&self) -> BoundaryCondition {
        BoundaryCondition::Inflow
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::SphericalPolar
    }

    fn end_time(&self) -> Option<f64> {
        Some(1.0)
    }
}

/// A relativistic shell is launched into a homologous, relativistic envelope.
pub struct EnvelopeShock {
    u_shell: f64,
    d_shell: f64,
    t_shell: f64,
    t_start: f64,
    r_inner: f64,
    r_outer: f64,
    expand: bool,
    form: kind_config::Form,
}

impl FromStr for EnvelopeShock {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        #[rustfmt::skip]
        let form = kind_config::Form::new()
            .item("u_shell",    30.0, "gamma-beta of the launched shell")
            .item("d_shell",    0.0,  "density enhancement (additive) of the launched shell")
            .item("t_shell",    1.0,  "time when the shell is launched from r=0")
            .item("t_start",    2.0,  "time when the simulation starts")
            .item("r_inner",    0.1,  "inner radius at start")
            .item("r_outer",   10.0,  "outer radius at start")
            .item("expand",    true,  "whether to expand the mesh homologously")
            .merge_string_args_allowing_duplicates(parameters.split(':').filter(|s| !s.is_empty()))
            .map_err(|e| InvalidSetup(format!("{}", e)))?;

        let setup = Self {
            u_shell: form.get("u_shell").into(),
            d_shell: form.get("d_shell").into(),
            t_shell: form.get("t_shell").into(),
            t_start: form.get("t_start").into(),
            r_inner: form.get("r_inner").into(),
            r_outer: form.get("r_outer").into(),
            expand: form.get("expand").into(),
            form,
        };

        if false {
            panic!()
        } else if setup.u_shell < 0.0 {
            Err(InvalidSetup(format!("u_shell must be >=0.0")))
        } else if setup.d_shell < 0.0 {
            Err(InvalidSetup(format!("d_shell must be >=0.0")))
        } else if setup.r_inner >= setup.r_outer {
            Err(InvalidSetup(format!("r_inner must be <r_outer")))
        } else {
            Ok(setup)
        }
    }
}

impl EnvelopeShock {
    fn r_shell(&self) -> f64 {
        self.t_start - self.t_shell
    }

    fn a0(&self) -> f64 {
        if self.expand {
            0.0
        } else {
            1.0
        }
    }

    fn a_dot(&self) -> f64 {
        if self.expand {
            1.0 / self.t_start
        } else {
            0.0
        }
    }

    fn num_decades(&self) -> f64 {
        (self.r_outer / self.r_inner).log10()
    }
}

impl Setup for EnvelopeShock {
    fn print_parameters(&self) {
        for key in self.form.sorted_keys() {
            println!(
                "{:.<20} {:<10} {}",
                key,
                self.form.get(&key),
                self.form.about(&key)
            );
        }
        println!("r_shell ....... {}", self.r_shell());
        println!("a_dot ......... {}", self.a_dot());
    }

    fn model_parameter_string(&self) -> String {
        self.form
            .iter()
            .map(|(a, b)| format!("{}={}", a, b))
            .collect::<Vec<_>>()
            .join(":")
    }

    fn num_primitives(&self) -> usize {
        3
    }

    fn solver_name(&self) -> String {
        "sr1d".to_owned()
    }

    fn initial_primitive(&self, r: f64, _q: f64, primitive: &mut [f64]) {
        let t = self.t_start;
        let r_shell = self.r_shell();
        let w_shell = self.r_shell() * 0.1;
        let d_shell = self.d_shell;
        let u_shell = self.u_shell;

        let prof = |r: f64| {
            if r > r_shell {
                0.0
            } else {
                f64::exp((r - r_shell) / w_shell)
            }
        };

        let ambient = RelativisticEnvelope {
            envelope_m1: 1.0,
            envelope_fastest_beta: 0.999,
            envelope_slowest_beta: 0.00,
            envelope_psi: 0.25,
            wind_mdot: 100.0,
        };

        let mdot = ambient.mass_rate_per_steradian(r, t);
        let u_ambient = ambient.gamma_beta(r, t);
        let d_ambient = mdot / (u_ambient * r * r);
        let p = 1e-5 * d_ambient;

        primitive[0] = d_ambient + d_shell * prof(r);
        primitive[1] = u_ambient + u_shell * prof(r);
        primitive[2] = p;
    }

    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 4.0 / 3.0,
        }
    }

    fn mesh(&self, resolution: u32) -> Mesh {
        Mesh::logarithmic_radial(self.r_inner, self.num_decades(), resolution)
    }

    fn homologous_mesh(&self) -> Option<(f64, f64)> {
        Some((self.a0(), self.a_dot()))
    }

    fn coordinate_system(&self) -> Coordinates {
        Coordinates::SphericalPolar
    }

    fn initial_time(&self) -> f64 {
        self.t_start
    }
}

pub struct RelativisticEnvelope {
    /// Mass coordinate of the u=1 shell
    pub envelope_m1: f64,

    /// Beta (v/c) of the slowest envelope shell
    pub envelope_slowest_beta: f64,

    /// Beta (v/c) of the outer shell
    pub envelope_fastest_beta: f64,

    /// Index psi in u(m) ~ m^-psi
    pub envelope_psi: f64,

    /// The mass loss rate for the wind
    pub wind_mdot: f64,
}

pub enum RelativisticEnvelopeZone {
    Envelope,
    Wind,
}

impl RelativisticEnvelope {
    pub fn envelop_slowest_u(&self) -> f64 {
        let b = self.envelope_slowest_beta;
        b / (1.0 - b * b).sqrt()
    }

    pub fn zone(&self, r: f64, t: f64) -> RelativisticEnvelopeZone {
        let v_min = self.envelope_slowest_beta;
        let r_wind_envelop_interface = v_min * t;

        if r > r_wind_envelop_interface {
            RelativisticEnvelopeZone::Envelope
        } else {
            RelativisticEnvelopeZone::Wind
        }
    }

    pub fn gamma_beta(&self, r: f64, t: f64) -> f64 {
        match self.zone(r, t) {
            RelativisticEnvelopeZone::Wind => self.envelop_slowest_u(),
            RelativisticEnvelopeZone::Envelope => {
                let b = f64::min(r / t, self.envelope_fastest_beta);
                let u = b / f64::sqrt(1.0 - b * b);
                u
            }
        }
    }

    pub fn mass_rate_per_steradian(&self, r: f64, t: f64) -> f64 {
        match self.zone(r, t) {
            RelativisticEnvelopeZone::Wind => self.wind_mass_rate_per_steradian(),
            RelativisticEnvelopeZone::Envelope => {
                let y = self.envelope_psi;
                let s = f64::min(r / t, self.envelope_fastest_beta);
                let f = f64::powf(s, -1.0 / y) * f64::powf(1.0 - s * s, 0.5 / y - 1.0);
                self.envelope_m1 / (4.0 * PI * y * t) * f
            }
        }
    }

    pub fn wind_mass_rate_per_steradian(&self) -> f64 {
        self.wind_mdot
    }
}
