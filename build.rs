fn main() {






    // OLD
    cc::Build::new()
        .file("src/physics/hydro.c")
        .flag("-Xpreprocessor")
        .flag("-fopenmp")
        .compile("hydro_omp");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .compile("hydro_cpu");

    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .file("src/physics/hydro.c")
            .cuda(true)
            .flag("-x=cu")
            .compile("hydro_gpu");

        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(not(feature = "cuda"))]
    {
        cc::Build::new()
            .file("src/physics/hydro.c")
            .define("GPU_STUBS", None)
            .flag("-Wno-unused-function")
            .compile("hydro_gpu");
    }



    // NEW
    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .file("src/solver/patch.c")
            .define("PATCH_LINKAGE", "extern \"C\"")
            .cuda(true)
            .flag("-x=cu")
            .compile("patch");
    }
    #[cfg(not(feature = "cuda"))]
    {
        cc::Build::new()
            .file("src/solver/patch.c")
            .define("PATCH_LINKAGE", None)
            .compile("patch");        
    }

    cc::Build::new()
        .file("src/solver/iso2d.c")
        .define("API_MODE_CPU", None)
        .compile("iso2d_cpu");

    #[cfg(feature = "omp")]
    {
        cc::Build::new()
            .file("src/solver/iso2d.c")
            .define("API_MODE_OMP", None)
            .flag("-Xpreprocessor")
            .flag("-fopenmp")
            .compile("iso2d_omp");
    }

    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .file("src/solver/iso2d.c")
            .cuda(true)
            .define("API_MODE_GPU", None)
            .flag("-x=cu")
            .compile("iso2d_gpu");
    }
}
