fn main() {
    cc::Build::new()
        .file("src/lib.cu")
        .cuda(true)
        .compile("lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
