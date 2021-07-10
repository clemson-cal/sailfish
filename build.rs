use cfg_if::cfg_if;

fn use_omp() -> bool {
    let use_omp;
    cfg_if! {
        if #[cfg(feature = "omp")] {
            use_omp = true;
        } else {
            use_omp = false;
        }
    }
    use_omp
}

fn main() {
    let plat = sf_build::Platform::discover();
    plat.build_src("src/iso2d/mod", use_omp())
        .compile("iso2d_mod");
    plat.emit_link_flags();
}
