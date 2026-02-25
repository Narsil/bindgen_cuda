fn main() {
    let builder = bindgen_cuda::Builder::default();
    builder.build_lib("libsin.a");
    println!("cargo:rustc-link-lib=sin");
    println!("cargo:rustc-link-search=native={}", ".");
}
