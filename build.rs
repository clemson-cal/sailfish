fn main() {
    cc::Build::new()
        .file("src/physics/hydro.c")
        .flag("-Xpreprocessor")
        .flag("-fopenmp")
        .compile("hydro");
}
