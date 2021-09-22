use anyhow::Result;
use clap::{AppSettings, Clap};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::ops::Range;
use std::path::Path;

type Rectangle<T> = (Range<T>, Range<T>);

#[derive(serde::Deserialize)]
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

#[derive(serde::Deserialize)]
struct Patch {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
    rect: Rectangle<i64>,
    num_fields: usize,
}

#[derive(serde::Deserialize)]
struct State {
    primitive_patches: Vec<Patch>,
    mesh: StructuredMesh,
}

impl State {
    fn load(filename: &str) -> Result<Self> {
        let mut file = File::open(filename)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(rmp_serde::from_read_ref(&bytes)?)
    }
    fn num_fields(&self) -> Result<usize> {
        Ok(self
            .primitive_patches
            .first()
            .ok_or(anyhow::anyhow!("empty patch list"))?
            .num_fields)
    }
}

struct Process {
    first_call: bool,
}

impl Process {
    fn new() -> Self {
        Self { first_call: true }
    }
}

struct Scaling {
    vmin: f64,
    vmax: f64,
    index: Option<f64>,
    log: bool,
}

impl Scaling {
    fn linear(vmin: f64, vmax: f64) -> Self {
        Self {
            vmin,
            vmax,
            index: None,
            log: false,
        }
    }
    fn log(vmin: f64, vmax: f64) -> Self {
        Self {
            vmin,
            vmax,
            index: None,
            log: true,
        }
    }
    fn plaw(vmin: f64, vmax: f64, index: f64) -> Self {
        Self {
            vmin,
            vmax,
            index: Some(index),
            log: false,
        }
    }
    fn scale(&self, x: f64) -> f64 {
        let y = if self.log {
            x.log10()
        } else if let Some(index) = self.index {
            x.powf(index)
        } else {
            x
        };
        (y - self.vmin) / (self.vmax - self.vmin)
    }
}

fn float_to_rgba(x: f64) -> [u8; 4] {
    let c = (x * 255.0) as u8;
    [c, c, c, 255]
}

fn sorted_field(filename: &str, field: usize) -> Result<Vec<f64>> {
    let state = State::load(filename)?;
    let mut data = vec![];
    for patch in &state.primitive_patches {
        for b in patch.data.chunks_exact(8 * state.num_fields()?) {
            let b = &b[field * 8..(field + 1) * 8];
            let bytes = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
            let x = f64::from_le_bytes(bytes);
            if !x.is_finite() {
                anyhow::bail!("field contains nan or inf")
            }
            data.push(x)
        }
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(data)
}

fn print_quantiles(filename: &str, field: usize) -> Result<()> {
    let data = sorted_field(filename, field)?;
    println!("quantiles for {}:", filename);
    println!("    max                {:+.03e}", data.last().unwrap());
    for x in (1..5).rev() {
        let y = libm::erf(x as f64 / f64::sqrt(2.0));
        let i = (data.len() as f64 * y).floor() as usize;
        println!("    +{} sigma ({:>6.3}%) {:+.03e}", x, y * 100.0, data[i]);
    }
    for x in 1..5 {
        let y = 1.0 - libm::erf(x as f64 / f64::sqrt(2.0));
        let i = (data.len() as f64 * y).ceil() as usize;
        println!("    -{} sigma ({:>6.3}%) {:+.03e}", x, y * 100.0, data[i]);
    }
    println!("    min                {:+.03e}", data.first().unwrap());
    Ok(())
}

fn make_image(
    filename: &str,
    field_index: usize,
    scaling: &Scaling,
    process: &mut Process,
) -> Result<()> {
    let state = State::load(filename)?;
    let ni = state.mesh.ni as usize;
    let nj = state.mesh.nj as usize;
    let num_fields = state.num_fields()?;

    if field_index >= num_fields {
        anyhow::bail!("invalid field index {}/{}", field_index, num_fields)
    }
    if process.first_call {
        println!("mesh shape is [{}, {}]", ni, nj);
        println!("there are {} patches", state.primitive_patches.len());
        println!("reading field {}/{}", field_index, num_fields);
    }

    let mut rgba_data = vec![0; ni * nj * 4];
    for patch in &state.primitive_patches {
        let i0 = patch.rect.0.start as usize;
        let j0 = patch.rect.1.start as usize;
        let i1 = patch.rect.0.end as usize;
        let j1 = patch.rect.1.end as usize;
        let sq = 8;
        let sj = 8 * num_fields;
        let si = 8 * num_fields * (j1 - j0);
        for i in i0..i1 {
            for j in j0..j1 {
                let n = (i - i0) * si + (j - j0) * sj + field_index * sq;
                let mut bytes = [0; 8];
                for (a, b) in bytes.iter_mut().zip(&patch.data[n..n + 8]) {
                    *a = *b
                }
                let x = f64::from_le_bytes(bytes);
                let y = scaling.scale(x);
                let c = float_to_rgba(y);
                let m = (i + (nj - 1 - j)  * ni) * 4;
                rgba_data[m..m + 4].copy_from_slice(&c);
            }
        }
    }

    let png_name = filename.replace(".sf", ".png");
    let path = Path::new(&png_name);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, ni as u32, nj as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_trns(vec![0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]);
    println!("write {}", png_name);

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&rgba_data).unwrap();

    process.first_call = false;
    Ok(())
}

#[derive(Clap)]
#[clap(version = "1.0", author = "J. Zrake <jzrake@clemson.edu>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    /// Location to read checkpoint files from
    paths: Vec<String>,

    /// The field index to read
    #[clap(long, short, default_value = "0")]
    field: usize,

    /// Just print the data quantiles and exit
    #[clap(long, short = 'q')]
    show_quantiles: bool,

    /// Scaling rule: [log|linear|plaw:n]
    #[clap(long, short, default_value = "linear")]
    scaling: String,

    /// The minimum data value ()
    #[clap(long, default_value = "0.0")]
    vmin: f64,

    /// The maximum data value
    #[clap(long, default_value = "1.0")]
    vmax: f64,
}

fn main() -> Result<()> {
    let opts: Opts = Opts::parse();
    let scaling = if opts.scaling == "linear" {
        Scaling::linear(opts.vmin, opts.vmax)
    } else if opts.scaling == "log" {
        Scaling::log(opts.vmin, opts.vmax)        
    } else if opts.scaling.starts_with("plaw:") {
        let index = opts.scaling[5..].parse()?;
        Scaling::plaw(opts.vmin, opts.vmax, index)
    } else {
        anyhow::bail!("scaling must be [log|linear|plaw:n]")
    };

    let mut process = Process::new();

    for filename in opts.paths {
        if opts.show_quantiles {
            print_quantiles(&filename, opts.field)?;
        } else {
            make_image(&filename, opts.field, &scaling, &mut process)?;
        }
    }
    Ok(())
}
