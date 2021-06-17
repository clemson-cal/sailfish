fn main() {
    cc::Build::new()
        .file("src/physics/hydro.c")
        .compile("hydro");
}
