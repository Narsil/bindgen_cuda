// Integration test for CUBIN bindings format verification
// This test is ignored on macOS as it requires CUDA toolkit

use std::env;
use std::fs;
use std::path::PathBuf;

#[test]
#[cfg_attr(target_os = "macos", ignore)]
fn test_cubin_bindings_format() {
    // This test verifies CUBIN generates correct binding types
    let temp_dir = env::temp_dir().join("bindgen_cuda_test_cubin_format");
    fs::create_dir_all(&temp_dir).unwrap();
    
    env::set_var("OUT_DIR", temp_dir.to_str().unwrap());
    
    let test_kernel_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_kernel.cu");
    let test_kernel_dst = temp_dir.join("test_kernel.cu");
    fs::copy(&test_kernel_src, &test_kernel_dst).unwrap();
    
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![test_kernel_dst])
        .out_dir(&temp_dir);
    let bindings = builder.build_cubin().unwrap();
    let bindings_path = temp_dir.join("bindings.rs");
    bindings.write(&bindings_path).unwrap();
    
    let bindings_content = fs::read_to_string(&bindings_path).unwrap();
    
    // CUBIN should use &[u8] and include_bytes!
    assert!(bindings_content.contains("&[u8]"), 
            "CUBIN bindings should use &[u8] type");
    assert!(bindings_content.contains("include_bytes!"), 
            "CUBIN bindings should use include_bytes! macro");
    assert!(!bindings_content.contains("&str"), 
            "CUBIN bindings should not use &str type");
    assert!(!bindings_content.contains("include_str!"), 
            "CUBIN bindings should not use include_str! macro");
    
    fs::remove_dir_all(&temp_dir).ok();
}
