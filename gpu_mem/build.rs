fn main() {
    use sf_build::Platform;
    let plat = Platform::discover();

    if !std::matches!(plat, Platform::Cuda | Platform::Rocm) {
        panic!("neither nvcc nor hipcc is installed");
    }
    plat.build().file("src/lib.cu").compile("lib");
}
