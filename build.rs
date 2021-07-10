use cfg_if::cfg_if;

fn use_omp() -> bool {
    cfg_if! {
        if #[cfg(feature = "omp")] {
            true
        } else {
            false
        }
    }
}

fn main() {
    let plat = sf_build::Platform::discover();
    plat.build_src("src/iso2d/mod", use_omp())
        .compile("iso2d_mod");
    plat.build_src("src/euler1d/mod", use_omp())
        .compile("euler1d_mod");
    plat.emit_link_flags();
}
