use cfg_if::cfg_if;

fn main() {
    cfg_if! {
        if #[cfg(all(feature = "omp", feature = "cuda"))] {
            cc::Build::new()
                .file("src/iso2d/mod.c")
                .cuda(true)
                .flag("-x=cu")
                .flag("-Xcompiler")
                .flag("-fopenmp")
                .compile("iso2d_mod");
        } else if #[cfg(all(feature = "omp", not(feature = "cuda")))] {
            cc::Build::new()
                .file("src/iso2d/mod.c")
                .flag("-Xpreprocessor")
                .flag("-fopenmp")
                .compile("iso2d_mod");
        } else if #[cfg(all(not(feature = "omp"), feature = "cuda"))] {
            cc::Build::new()
                .file("src/iso2d/mod.c")
                .cuda(true)
                .flag("-x=cu")
                .compile("iso2d_mod");
        } else {
            cc::Build::new()
                .file("src/iso2d/mod.c")
                .compile("iso2d_mod");
        }
    }
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            println!("cargo:rustc-link-lib=dylib=cudart");            
        }
    }
}
