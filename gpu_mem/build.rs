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

fn gpu_build(src: &str) -> cc::Build {
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

fn gpu_link_flags() {
    if is_program_in_path("nvcc") {
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else if is_program_in_path("hipcc") {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    } else {
        panic!("neither nvcc nor hipcc is installed");        
    }
}

fn main() {
    gpu_build("src/lib.cu").compile("lib");
    gpu_link_flags();
}
