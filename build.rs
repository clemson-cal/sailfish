fn main() {
    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("SINGLE", None)
        .flag("-Wno-unknown-pragmas")
        .compile("physics_cpu_f32");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("DOUBLE", None)
        .flag("-Wno-unknown-pragmas")
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
}
