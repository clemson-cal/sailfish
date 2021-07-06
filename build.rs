use cfg_if::cfg_if;

fn compile_c_mod(mod_name: &str) {
    let src = format!("src/{}/mod.c", mod_name);
    let lib = format!("{}_mod", mod_name);

    cfg_if! {
        if #[cfg(all(feature = "omp", feature = "cuda"))] {
            cc::Build::new()
                .file(&src)
                .cuda(true)
                .flag("-x=cu")
                .flag("-Xcompiler")
                .flag("-fopenmp")
                .compile(&lib);
        } else if #[cfg(all(feature = "omp", not(feature = "cuda")))] {
            cc::Build::new()
                .file(&src)
                .flag("-Xpreprocessor")
                .flag("-fopenmp")
                .compile(&lib);
        } else if #[cfg(all(not(feature = "omp"), feature = "cuda"))] {
            cc::Build::new()
                .file(&src)
                .cuda(true)
                .flag("-x=cu")
                .compile(&lib);
        } else {
            cc::Build::new()
                .file(&src)
                .compile(&lib);
        }
    }    
}

fn main() {
    compile_c_mod("iso2d");
    compile_c_mod("euler1d");
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            println!("cargo:rustc-link-lib=dylib=cudart");            
        }
    }
}
