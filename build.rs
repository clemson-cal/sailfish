fn main() {

    cc::Build::new()
        .file("src/physics/hydro.c")
        .flag("-Xpreprocessor")
        .flag("-fopenmp")
        .compile("hydro_omp");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .compile("hydro_cpu");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("GPU_STUBS", None)
        .flag("-Wno-unused-function")
        .compile("hydro_gpu");

    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-lib=cuda");
}
