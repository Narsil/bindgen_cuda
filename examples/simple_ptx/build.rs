fn main() {
    let builder = bindgen_cuda::Builder::default();
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/kernel.rs").unwrap();
}
