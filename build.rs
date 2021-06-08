fn main() {
    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("SINGLE", None)
        .compile("physics_f32.a");

    cc::Build::new()
        .file("src/physics/hydro.c")
        .define("DOUBLE", None)
        .compile("physics_f64.a");
}
