use std::env;
use std::fs;

fn is_program_in_path(program: &str) -> bool {
    if let Ok(path) = env::var("PATH") {
        for p in path.split(":") {
            let p_str = format!("{}/{}", p, program);
            if fs::metadata(p_str).is_ok() {
                return true;
            }
        }
    }
    false
}

pub enum Platform {
    NoGpu,
    Cuda,
    Rocm,
}

impl Platform {
    pub fn discover() -> Self {
        if is_program_in_path("nvcc") {
            Platform::Cuda
        } else if is_program_in_path("hipcc") {
            Platform::Rocm
        } else {
            Platform::NoGpu
        }
    }

    pub fn emit_link_flags(&self) {
        match self {
            Platform::NoGpu => {}
            Platform::Cuda => {
                println!("cargo:rustc-link-lib=dylib=cudart");
            }
            Platform::Rocm => {
                println!("cargo:rustc-link-search=native=/opt/rocm/lib");
                println!("cargo:rustc-link-lib=dylib=amdhip64");
            }
        }
    }

    pub fn build(&self) -> cc::Build {
        match self {
            Platform::NoGpu => cc::Build::new().flag("-std=c99").clone(),
            Platform::Cuda => cc::Build::new().cuda(true).clone(),
            Platform::Rocm => cc::Build::new()
                .compiler("hipcc")
                .define("__ROCM__", None)
                .clone(),
        }
    }

    pub fn build_omp(&self) -> cc::Build {
        match self {
            Platform::NoGpu => self.build().flag("-Xpreprocessor").flag("-fopenmp").clone(),
            Platform::Cuda => self.build().flag("-Xcompiler").flag("-fopenmp").clone(),
            Platform::Rocm => panic!("ROCm build is not compatible with OpenMP"),
        }
    }

    pub fn build_src(&self, src: &str, use_omp: bool) -> cc::Build {
        let src = if std::matches!(self, Platform::Cuda | Platform::Rocm) {
            src.to_owned() + ".cu"
        } else {
            src.to_owned() + ".c"
        };
        let mut build = if use_omp {
            self.build_omp()
        } else {
            self.build()
        };
        build.file(src).clone()
    }
}
