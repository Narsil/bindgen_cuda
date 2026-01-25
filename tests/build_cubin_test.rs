// Integration test for bindgen_cuda CUBIN build functionality
// This test is ignored on macOS as it requires CUDA toolkit

use std::env;
use std::fs;
use std::path::PathBuf;

#[test]
#[cfg_attr(target_os = "macos", ignore)]
fn test_build_cubin() {
    // Setup test environment
    let temp_dir = env::temp_dir().join("bindgen_cuda_test_cubin");
    fs::create_dir_all(&temp_dir).unwrap();
    
    // Set OUT_DIR for the builder
    env::set_var("OUT_DIR", temp_dir.to_str().unwrap());
    
    // Copy test kernel to temp directory
    let test_kernel_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_kernel.cu");
    let test_kernel_dst = temp_dir.join("test_kernel.cu");
    fs::copy(&test_kernel_src, &test_kernel_dst).unwrap();
    
    // Build CUBIN
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![test_kernel_dst.clone()])
        .out_dir(&temp_dir);
    
    let bindings = builder.build_cubin().expect("Failed to build CUBIN");
    
    // Write bindings
    let bindings_path = temp_dir.join("cubin_bindings.rs");
    bindings.write(&bindings_path).expect("Failed to write CUBIN bindings");
    
    // Verify CUBIN file was created
    let cubin_file = temp_dir.join("test_kernel.cubin");
    assert!(cubin_file.exists(), "CUBIN file should be created");
    
    // Verify bindings file was created and contains expected content
    assert!(bindings_path.exists(), "Bindings file should be created");
    let bindings_content = fs::read_to_string(&bindings_path).unwrap();
    assert!(bindings_content.contains("pub const TEST_KERNEL"), 
            "Bindings should contain TEST_KERNEL constant");
    assert!(bindings_content.contains("&[u8]"), 
            "CUBIN bindings should use &[u8] type");
    assert!(bindings_content.contains("include_bytes!"), 
            "CUBIN bindings should use include_bytes! macro");
    
    // Verify CUBIN content is binary (ELF format)
    let cubin_content = fs::read(&cubin_file).unwrap();
    assert!(cubin_content.len() > 4, "CUBIN file should not be empty");
    // Check for ELF magic number: 0x7f 'E' 'L' 'F'
    assert_eq!(cubin_content[0], 0x7f, "CUBIN should start with ELF magic number");
    assert_eq!(cubin_content[1], b'E', "CUBIN should be ELF format");
    assert_eq!(cubin_content[2], b'L', "CUBIN should be ELF format");
    assert_eq!(cubin_content[3], b'F', "CUBIN should be ELF format");
    
    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}
