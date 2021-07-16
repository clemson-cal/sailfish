use crate::error::{self, Error::*};
use crate::lookup_table::LookupTable;
use crate::mesh::Mesh;
use crate::sailfish::{BufferZone, Coordinates, EquationOfState, PointMass, SinkModel, StructuredMesh};
use kepler_two_body::{OrbitalElements, OrbitalState};

pub trait Setup {
    fn print_parameters(&self);
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]);
    fn initial_time(&self) -> f64 {
        0.0
    }
    fn end_time(&self) -> Option<f64> {
        None
    }
    fn masses(&self, time: f64) -> Vec<PointMass>;
    fn equation_of_state(&self) -> EquationOfState;
    fn buffer_zone(&self) -> BufferZone;
    fn viscosity(&self) -> Option<f64>;
    fn mesh(&self, resolution: u32) -> Mesh;
    fn coordinate_system(&self) -> Coordinates;
    fn initial_primitive_vec(&self, mesh: &Mesh) -> Vec<f64> {
        match mesh {
            Mesh::Structured(mesh) => {
                let mut primitive = vec![0.0; ((mesh.ni + 4) * (mesh.nj + 4) * 3) as usize];
                let si = 3 * (mesh.nj + 4);
                let sj = 3;
                for i in -2i32..mesh.ni + 2 {
                    for j in -2i32..mesh.nj + 2 {
                        let n = ((i + 2) * si + (j + 2) * sj) as usize;
                        let [x, y] = mesh.cell_coordinates(i, j);
                        self.initial_primitive(x, y, &mut primitive[n..n + 3])
                    }
                }
                primitive
            }
            Mesh::FacePositions1D(faces) => {
                let mut primitive = vec![0.0; (faces.len() - 1) * 3];
                for i in 0..faces.len() - 1 {
                    let x = 0.5 * (faces[i] + faces[i + 1]);
                    self.initial_primitive(x, 0.0, &mut primitive[3 * i..3 * i + 3]);
                }
                primitive
            }
        }
    }
}

pub struct Explosion {}

impl std::str::FromStr for Explosion {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self {})
        } else {
            Err(InvalidSetup(format!(
                "explosion problem does not take any parameters, got {}",
                parameters
            )))
        }
    }
}

impl Setup for Explosion {
    fn print_parameters(&self) {}
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
        EquationOfState::Isothermal {
            sound_speed_squared: 1.0,
        }
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::NoBuffer
    }
    fn viscosity(&self) -> Option<f64> {
        None
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
    pub sink_radius: f64,
    pub sink_rate: f64,
    pub sink_model: SinkModel,
    form: kind_config::Form,
}

impl std::str::FromStr for Binary {
    type Err = error::Error;

    #[rustfmt::skip]
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        let form = kind_config::Form::new()
            .item("domain_radius", 12.0, "half-size of the simulation domain [a]")
            .item("nu",            1e-3, "kinematic viscosity coefficient [Omega a^2]")
            .item("mach_number",   10.0, "mach number for locally isothermal EOS")
            .item("sink_radius",   0.05, "sink kernel radius [a]")
            .item("sink_rate",     10.0, "rate of mass subtraction in the sink [Omega]")
            .item("sink_model",    "af", "sink prescription: [none|af|tf|ff]")
            .merge_string_args_allowing_duplicates(parameters.split(':').filter(|s| !s.is_empty()))
            .map_err(|e| InvalidSetup(format!("{}", e)))?;

        Ok(Self {
            domain_radius: form.get("domain_radius").into(),
            nu: form.get("nu").into(),
            mach_number: form.get("mach_number").into(),
            sink_radius: form.get("sink_radius").into(),
            sink_rate: form.get("sink_rate").into(),
            sink_model: match form.get("sink_model").to_string().as_str() {
                "none" => SinkModel::Inactive,
                "af" => SinkModel::AccelerationFree,
                "tf" => SinkModel::TorqueFree,
                "ff" => SinkModel::ForceFree,
                _ => return Err(InvalidSetup("invalid sink_model".into())),
            },
            form,
        })
    }
}

impl Setup for Binary {
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

    #[allow(clippy::many_single_char_names)]
    fn initial_primitive(&self, x: f64, y: f64, primitive: &mut [f64]) {
        let r = (x * x + y * y).sqrt();
        let rs = (x * x + y * y + self.sink_radius.powf(2.0)).sqrt();
        let phi_hat_x = -y / r.max(1e-12);
        let phi_hat_y = x / r.max(1e-12);
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
            vx: mass0.velocity_x(),
            vy: mass0.velocity_y(),
            mass: mass0.mass(),
            rate: self.sink_rate,
            radius: self.sink_radius,
            model: self.sink_model,
        };
        let mass1 = PointMass {
            x: mass1.position_x(),
            y: mass1.position_y(),
            vx: mass1.velocity_x(),
            vy: mass1.velocity_y(),
            mass: mass1.mass(),
            rate: self.sink_rate,
            radius: self.sink_radius,
            model: self.sink_model,
        };
        vec![mass0, mass1]
    }
    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::LocallyIsothermal {
            mach_number_squared: self.mach_number.powi(2),
        }
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::NoBuffer
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

pub struct Shocktube {}

impl std::str::FromStr for Shocktube {
    type Err = error::Error;
    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        if parameters.is_empty() {
            Ok(Self {})
        } else {
            Err(InvalidSetup(format!(
                "shocktube problem does not take any parameters, got {}",
                parameters
            )))
        }
    }
}

impl Setup for Shocktube {
    fn print_parameters(&self) {}
    fn initial_primitive(&self, x: f64, _y: f64, primitive: &mut [f64]) {
        if x < 0.5 {
            primitive[0] = 1.0;
            primitive[2] = 1.0;
        } else {
            primitive[0] = 0.1;
            primitive[2] = 0.125;
        }
    }
    fn masses(&self, _time: f64) -> Vec<PointMass> {
        vec![]
    }
    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 5.0 / 3.0,
        }
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::NoBuffer
    }
    fn viscosity(&self) -> Option<f64> {
        None
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

pub struct Sedov {
    faces: Vec<f64>,
    table: LookupTable<4>,
}

impl std::str::FromStr for Sedov {
    type Err = error::Error;

    fn from_str(parameters: &str) -> Result<Self, Self::Err> {
        use std::iter::once;

        let filename = parameters;
        let table = LookupTable::<4>::from_ascii_file(filename)
            .map_err(|e| InvalidSetup(format!("{}", e)))?;

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
        Ok(Self { faces, table })
    }
}

impl Setup for Sedov {
    fn print_parameters(&self) {}
    fn initial_primitive(&self, x: f64, _y: f64, primitive: &mut [f64]) {
        let row = self.table.sample(x);
        primitive[0] = row[1].max(1e-4);
        primitive[1] = row[2];
        primitive[2] = row[3].max(1e-10);
    }
    fn masses(&self, _time: f64) -> Vec<PointMass> {
        vec![]
    }
    fn equation_of_state(&self) -> EquationOfState {
        EquationOfState::GammaLaw {
            gamma_law_index: 5.0 / 3.0,
        }
    }
    fn buffer_zone(&self) -> BufferZone {
        BufferZone::NoBuffer
    }
    fn viscosity(&self) -> Option<f64> {
        None
    }
    fn mesh(&self, _resolution: u32) -> Mesh {
        // Note: resolution is ignored. Consider making it an Option, and
        // returning Result in case it's given for problems that specify the
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
