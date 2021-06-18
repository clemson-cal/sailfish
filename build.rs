fn main() {

    #[cfg(feature = "omp")]
    cc::Build::new()
        .file("src/physics/hydro.c")
        .flag("-Xpreprocessor")
        .flag("-fopenmp")
        .compile("hydro");

    #[cfg(not(feature = "omp"))]
    cc::Build::new()
        .file("src/physics/hydro.c")
        .compile("hydro");
}
