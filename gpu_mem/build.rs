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

fn main() {
    if is_program_in_path("nvcc") {
        cc::Build::new()
            .file("src/lib.cu")
            .cuda(true)
            .flag("-x=cu")
            .compile("lib");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else if is_program_in_path("hipcc") {
        cc::Build::new()
            .file("src/lib.cu")
            .compiler("hipcc")
            .compile("lib");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    } else {
        panic!("nvcc is not installed");
    }
}
