fn main() {
    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("SINGLE", None)
        .compile("physics_cpu_f32");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("DOUBLE", None)
        .compile("physics_cpu_f64");

    #[cfg(feature="omp")]
    {
        cc::Build::new()
            .file("src/physics/hydro.c")
            .flag("-Xpreprocessor")
            .flag("-fopenmp")
            .define("SINGLE", None)
            .compile("physics_omp_f32");

        cc::Build::new()
            .file("src/physics/hydro.c")
            .flag("-Xpreprocessor")
            .flag("-fopenmp")
            .define("DOUBLE", None)
            .compile("physics_omp_f64");
    }

    #[cfg(feature="cuda")]
    {
        cc::Build::new()
            .file("src/physics/hydro.cu")
            .cuda(true)
            .compile("test");

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");        
    }
}
