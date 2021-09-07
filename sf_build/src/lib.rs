pub enum Platform {
    NoGpu,
    Cuda,
    Rocm,
}

impl Platform {
    pub fn discover(gpu: bool) -> Self {
        use which::which;

        if gpu && which("nvcc").is_ok() {
            Platform::Cuda
        } else if gpu && which("hipcc").is_ok() {
            Platform::Rocm
        } else {
            Platform::NoGpu
        }
    }

    pub fn emit_link_flags(&self) {
        match self {
            Platform::NoGpu => {}
            Platform::Cuda => {
                if let Ok(cuda_lib) = std::env::var("CUDA_LIB") {
                    println!("cargo:rustc-link-search=native={}", cuda_lib)
                }
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
                .cpp(true)
                .compiler("hipcc")
                .define("__ROCM__", None)
                .clone(),
        }
    }

    pub fn build_omp(&self) -> cc::Build {
        match self {
            Platform::NoGpu => self.build().flag("-Xpreprocessor").flag("-fopenmp").clone(),
            Platform::Cuda => self.build().flag("-Xcompiler").flag("-fopenmp").clone(),
            Platform::Rocm => self.build().flag("-fopenmp").clone(),
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
