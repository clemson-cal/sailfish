fn main() {
    let mut hydro = cc::Build::new();
    hydro.file("src/physics/hydro.c");
    #[cfg(feature = "omp")]
    hydro.flag("-Xpreprocessor").flag("-fopenmp");
    #[cfg(not(feature = "cuda"))]
    hydro.define("GPU_STUBS", None);
    hydro.compile("hydro");

    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .file("src/physics/hydro.cu")
            .cuda(true)
            .compile("hydro-cuda");

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    }
}
