use cfg_if::cfg_if;

fn main() {
    // #[cfg(feature = "cuda")]
    // {
    //     cc::Build::new()
    //         .file("src/solver/patch.c")
    //         .define("PATCH_LINKAGE", "extern \"C\"")
    //         .cuda(true)
    //         .flag("-x=cu")
    //         .compile("patch");
    // }
    // #[cfg(not(feature = "cuda"))]
    // {
    //     cc::Build::new()
    //         .file("src/solver/patch.c")
    //         .define("PATCH_LINKAGE", "")
    //         .compile("patch");
    // }

    // cc::Build::new()
    //     .file("src/solver/iso2d.c")
    //     .define("API_MODE_CPU", None)
    //     .flag("-Wno-unused-function")
    //     .compile("iso2d_cpu");

    // #[cfg(feature = "omp")]
    // {
    //     cc::Build::new()
    //         .file("src/solver/iso2d.c")
    //         .define("API_MODE_OMP", None)
    //         .flag("-Wno-unused-function")
    //         .flag("-Xpreprocessor")
    //         .flag("-fopenmp")
    //         .compile("iso2d_omp");
    // }

    // #[cfg(feature = "cuda")]
    // {
    //     cc::Build::new()
    //         .file("src/solver/iso2d.c")
    //         .cuda(true)
    //         .define("API_MODE_GPU", None)
    //         .flag("-x=cu")
    //         .compile("iso2d_gpu");
    // }

    // #[cfg(feature = "cuda")]
    // println!("cargo:rustc-link-lib=dylib=cudart");

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
