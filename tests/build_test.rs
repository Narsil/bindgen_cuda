// Integration test for bindgen_cuda build functionality
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

#[test]
#[cfg_attr(target_os = "macos", ignore)]
fn test_ptx_vs_cubin_bindings_difference() {
    // This test verifies that PTX and CUBIN generate different binding types
    let temp_dir_ptx = env::temp_dir().join("bindgen_cuda_test_compare_ptx");
    let temp_dir_cubin = env::temp_dir().join("bindgen_cuda_test_compare_cubin");
    
    fs::create_dir_all(&temp_dir_ptx).unwrap();
    fs::create_dir_all(&temp_dir_cubin).unwrap();
    
    let test_kernel_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_kernel.cu");
    
    // Build PTX
    env::set_var("OUT_DIR", temp_dir_ptx.to_str().unwrap());
    let test_kernel_ptx = temp_dir_ptx.join("test_kernel.cu");
    fs::copy(&test_kernel_src, &test_kernel_ptx).unwrap();
    
    let builder_ptx = bindgen_cuda::Builder::default()
        .kernel_paths(vec![test_kernel_ptx])
        .out_dir(&temp_dir_ptx);
    let bindings_ptx = builder_ptx.build_ptx().unwrap();
    let bindings_path_ptx = temp_dir_ptx.join("bindings.rs");
    bindings_ptx.write(&bindings_path_ptx).unwrap();
    
    // Build CUBIN
    env::set_var("OUT_DIR", temp_dir_cubin.to_str().unwrap());
    let test_kernel_cubin = temp_dir_cubin.join("test_kernel.cu");
    fs::copy(&test_kernel_src, &test_kernel_cubin).unwrap();
    
    let builder_cubin = bindgen_cuda::Builder::default()
        .kernel_paths(vec![test_kernel_cubin])
        .out_dir(&temp_dir_cubin);
    let bindings_cubin = builder_cubin.build_cubin().unwrap();
    let bindings_path_cubin = temp_dir_cubin.join("bindings.rs");
    bindings_cubin.write(&bindings_path_cubin).unwrap();
    
    // Compare bindings
    let ptx_bindings = fs::read_to_string(&bindings_path_ptx).unwrap();
    let cubin_bindings = fs::read_to_string(&bindings_path_cubin).unwrap();
    
    // PTX should use &str and include_str!
    assert!(ptx_bindings.contains("&str"));
    assert!(ptx_bindings.contains("include_str!"));
    assert!(!ptx_bindings.contains("&[u8]"));
    assert!(!ptx_bindings.contains("include_bytes!"));
    
    // CUBIN should use &[u8] and include_bytes!
    assert!(cubin_bindings.contains("&[u8]"));
    assert!(cubin_bindings.contains("include_bytes!"));
    assert!(!cubin_bindings.contains("&str"));
    assert!(!cubin_bindings.contains("include_str!"));
    
    // Cleanup
    fs::remove_dir_all(&temp_dir_ptx).ok();
    fs::remove_dir_all(&temp_dir_cubin).ok();
}
