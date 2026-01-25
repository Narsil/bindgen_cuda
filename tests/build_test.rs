// Integration test for bindgen_cuda PTX build functionality
// This test is ignored on macOS as it requires CUDA toolkit

use std::env;
use std::fs;
use std::path::PathBuf;

#[test]
#[cfg_attr(target_os = "macos", ignore)]
fn test_build_ptx() {
    // Setup test environment
    let temp_dir = env::temp_dir().join("bindgen_cuda_test_ptx");
    fs::create_dir_all(&temp_dir).unwrap();
    
    // Set OUT_DIR for the builder
    env::set_var("OUT_DIR", temp_dir.to_str().unwrap());
    
    // Copy test kernel to temp directory
    let test_kernel_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_kernel.cu");
    let test_kernel_dst = temp_dir.join("test_kernel.cu");
    fs::copy(&test_kernel_src, &test_kernel_dst).unwrap();
    
    // Build PTX
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![test_kernel_dst.clone()])
        .out_dir(&temp_dir);
    
    let bindings = builder.build_ptx().expect("Failed to build PTX");
    
    // Write bindings
    let bindings_path = temp_dir.join("ptx_bindings.rs");
    bindings.write(&bindings_path).expect("Failed to write PTX bindings");
    
    // Verify PTX file was created
    let ptx_file = temp_dir.join("test_kernel.ptx");
    assert!(ptx_file.exists(), "PTX file should be created");
    
    // Verify bindings file was created and contains expected content
    assert!(bindings_path.exists(), "Bindings file should be created");
    let bindings_content = fs::read_to_string(&bindings_path).unwrap();
    assert!(bindings_content.contains("pub const TEST_KERNEL"), 
            "Bindings should contain TEST_KERNEL constant");
    assert!(bindings_content.contains("&str"), 
            "PTX bindings should use &str type");
    assert!(bindings_content.contains("include_str!"), 
            "PTX bindings should use include_str! macro");
    
    // Verify PTX content is valid text
    let ptx_content = fs::read_to_string(&ptx_file).unwrap();
    assert!(ptx_content.contains(".version") || ptx_content.contains(".target"),
            "PTX file should contain valid PTX directives");
    
    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}


