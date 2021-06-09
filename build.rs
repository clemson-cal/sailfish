fn main() {
    println!("cargo:rustc-link-lib=omp");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .flag("-Xpreprocessor")
        .flag("-fopenmp")
        .define("SINGLE", None)
        .compile("physics_f32.a");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .flag("-Xpreprocessor")
        .flag("-fopenmp")
        .define("DOUBLE", None)
        .compile("physics_f64.a");
}
