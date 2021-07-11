fn main() {
    use sf_build::Platform;
    let plat = Platform::discover(true);

    if !std::matches!(plat, Platform::Cuda | Platform::Rocm) {
        println!("cargo:warning=neither nvcc nor hipcc is installed; linker would fail");
    } else {
        plat.build().file("src/lib.cu").compile("lib");
    }
}
