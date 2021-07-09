use std::env;
use std::fs;
use cfg_if::cfg_if;

#[allow(dead_code)]
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

#[allow(dead_code)]
fn gpu_build(src: &str) -> cc::Build {
    let src = src.to_owned() + "u";
    if is_program_in_path("nvcc") {
        cc::Build::new()
            .file(src)
            .cuda(true)
            .clone()
    } else if is_program_in_path("hipcc") {
        cc::Build::new()
            .file(src)
            .compiler("hipcc")
            .clone()
    } else {
        panic!("neither nvcc nor hipcc is installed");        
    }
}

fn build(src: &str) -> cc::Build {
    cfg_if! {
        if #[cfg(all(feature = "omp", feature = "cuda"))] {
            gpu_build(src)
                //.flag("-Xcompiler")
                .flag("-fopenmp")
                .clone()
        } else if #[cfg(all(feature = "omp", not(feature = "cuda")))] {
            cc::Build::new()
                .file(src)
                .flag("-Xpreprocessor")
                .flag("-fopenmp")
                .clone()
        } else if #[cfg(all(not(feature = "omp"), feature = "cuda"))] {
            gpu_build(src)
        } else {
            cc::Build::new()
                .file(src)
                .clone()
        }
    }
}

fn gpu_link_flags() {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            if is_program_in_path("nvcc") {
                println!("cargo:rustc-link-lib=dylib=cudart");
            } else if is_program_in_path("hipcc") {
                println!("cargo:rustc-link-search=native=/opt/rocm/lib");
                println!("cargo:rustc-link-lib=dylib=amdhip64");
            } else {
                panic!("neither nvcc nor hipcc is installed");        
            }
        }
    }
}

fn main() {
    build("src/iso2d/mod.c").compile("iso2d_mod");
    gpu_link_flags();
}
