use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum Error {}

#[derive(Default)]
pub struct Builder {
    cuda_root: Option<PathBuf>,
    kernel_paths: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    extra_args: Vec<&'static str>,
}

pub struct Bindings {
    write: bool,
    paths: Vec<PathBuf>,
}

fn default_kernels() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.cu")
            .ok()?
            .map(|p| p.expect("Invalid path"))
            .collect(),
    )
}
fn default_include() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.cuh")
            .ok()?
            .map(|p| p.expect("Invalid path"))
            .collect(),
    )
}

impl Builder {
    pub fn new() -> Self {
        let cuda_root = cuda_include_dir();
        let kernel_paths = default_kernels().unwrap_or(vec![]);
        let include_paths = default_include().unwrap_or(vec![]);
        let extra_args = vec![];
        Self {
            cuda_root,
            kernel_paths,
            include_paths,
            extra_args,
        }
    }

    pub fn kernel_paths_glob(&mut self, glob: &str) {
        self.kernel_paths = glob::glob(glob)
            .expect("Invalid blob")
            .map(|p| p.expect("Invalid path"))
            .collect();
    }

    pub fn include_paths_glob(&mut self, glob: &str) {
        self.include_paths = glob::glob(glob)
            .expect("Invalid blob")
            .map(|p| p.expect("Invalid path"))
            .collect();
    }

    pub fn cuda_root<P>(&mut self, path: P)
    where
        P: Into<PathBuf>,
    {
        self.cuda_root = Some(path.into());
    }

    pub fn build(self) -> Result<Bindings, Error> {
        println!("cargo:rerun-if-changed=build.rs");
        let cuda_root = self.cuda_root.expect("Could not find CUDA in standard locations, set it manually using Builder().set_cuda_root(...)");
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            cuda_root.join("include").display()
        );
        let out_dir = std::env::var("OUT_DIR").unwrap();

        let mut include_directories = self.include_paths;
        for path in &mut include_directories {
            // println!("cargo:rerun-if-changed={}", path.display());
            let destination =
                std::format!("{out_dir}/{}", path.file_name().unwrap().to_str().unwrap());
            std::fs::copy(path.clone(), destination).unwrap();
            // remove the filename from the path so it's just the directory
            path.pop();
        }

        include_directories.sort();
        include_directories.dedup();

        let compute_cap = compute_cap().expect("Could not get Cuda compute cap");

        #[allow(unused)]
        let include_options: Vec<String> = include_directories
            .into_iter()
            .map(|s| "-I".to_string() + &s.into_os_string().into_string().unwrap())
            .collect::<Vec<_>>();

        let ccbin_env = std::env::var("NVCC_CCBIN");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
        let children = self.kernel_paths
            // .par_iter()
            .iter()
            .flat_map(|p| {
                let mut output = p.clone();
                output.set_extension("ptx");
                let output_filename = std::path::Path::new(&out_dir).to_path_buf().join("out").with_file_name(output.file_name().unwrap());

                let ignore = if output_filename.exists() {
                    let out_modified = output_filename.metadata().unwrap().modified().unwrap();
                    let in_modified = p.metadata().unwrap().modified().unwrap();
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    false
                };
                if ignore {
                    None
                } else {
                    let mut command = std::process::Command::new("nvcc");
                    command.arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("--ptx")
                        .args(["--default-stream", "per-thread"])
                        .args(["--output-directory", &out_dir])
                        .args(&self.extra_args)
                        .args(&include_options);
                    if let Ok(ccbin_path) = &ccbin_env {
                        command
                            .arg("-allow-unsupported-compiler")
                            .args(["-ccbin", ccbin_path]);
                    }
                    command.arg(p);
                    Some((p, command.spawn()
                        .expect("nvcc failed to start. Ensure that you have CUDA installed and that `nvcc` is in your PATH.").wait_with_output()))
                }
            })
            .collect::<Vec<_>>();

        let ptx_paths: Vec<PathBuf> = glob::glob(&format!("{out_dir}/**/*.ptx"))
            .unwrap()
            .map(|p| p.unwrap())
            .collect();
        // We should rewrite `src/lib.rs` only if there are some newly compiled kernels, or removed
        // some old ones
        let write = !children.is_empty() || self.kernel_paths.len() < ptx_paths.len();
        for (kernel_path, child) in children {
            let output = child.expect("nvcc failed to run. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
            assert!(
                output.status.success(),
                "nvcc error while compiling {kernel_path:?}:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(Bindings {
            write,
            paths: self.kernel_paths,
        })
    }
}

impl Bindings {
    pub fn write<P>(&self, out: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        if self.write {
            let mut file = std::fs::File::create(out).unwrap();
            for kernel_path in &self.paths {
                let name = kernel_path.file_stem().unwrap().to_str().unwrap();
                file.write_all(
                format!(
                    r#"pub const {}: &str = include_str!(concat!(env!("OUT_DIR"), "/{}.ptx"));"#,
                    name.to_uppercase().replace('.', "_"),
                    name
                )
                .as_bytes(),
            )
            .unwrap();
                file.write_all(&[b'\n']).unwrap();
            }
        }
        Ok(())
    }
}

pub fn cuda_include_dir() -> Option<PathBuf> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    #[allow(unused)]
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    #[allow(unused)]
    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    #[cfg(feature = "ci-check")]
    let root: PathBuf = "ci".into();

    #[cfg(not(feature = "ci-check"))]
    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
}

pub fn build_ptx() -> (bool, Vec<std::path::PathBuf>) {
    // use rayon::prelude::*;
    let kernel_paths: Vec<PathBuf> = glob::glob("src/*.cu")
        .unwrap()
        .map(|p| p.unwrap())
        .collect();
    let mut include_directories: Vec<PathBuf> = glob::glob("src/**/*.cuh")
        .unwrap()
        .map(|p| p.unwrap())
        .collect();

    println!("cargo:rerun-if-changed=src/");
    // for path in &kernel_paths {
    //     println!("cargo:rerun-if-changed={}", path.display());
    // }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    for path in &mut include_directories {
        // println!("cargo:rerun-if-changed={}", path.display());
        let destination = std::format!("{out_dir}/{}", path.file_name().unwrap().to_str().unwrap());
        std::fs::copy(path.clone(), destination).unwrap();
        // remove the filename from the path so it's just the directory
        path.pop();
    }

    include_directories.sort();
    include_directories.dedup();

    let compute_cap = compute_cap().expect("Could not get Cuda compute cap");

    #[allow(unused)]
    let include_options: Vec<String> = include_directories
        .into_iter()
        .map(|s| "-I".to_string() + &s.into_os_string().into_string().unwrap())
        .collect::<Vec<_>>();

    let ccbin_env = std::env::var("NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    let children = kernel_paths
            //.par_iter()
            .iter()
            .flat_map(|p| {
                let mut output = p.clone();
                output.set_extension("ptx");
                let output_filename = std::path::Path::new(&out_dir).to_path_buf().join("out").with_file_name(output.file_name().unwrap());

                let ignore = if output_filename.exists() {
                    let out_modified = output_filename.metadata().unwrap().modified().unwrap();
                    let in_modified = p.metadata().unwrap().modified().unwrap();
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    false
                };
                if ignore {
                    None
                } else {
                    let mut command = std::process::Command::new("nvcc");
                    command.arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("--ptx")
                        .args(["--default-stream", "per-thread"])
                        .args(["--output-directory", &out_dir])
                        // Flash attention only
                        // .arg("--expt-relaxed-constexpr")
                        .args(&include_options);
                    if let Ok(ccbin_path) = &ccbin_env {
                        command
                            .arg("-allow-unsupported-compiler")
                            .args(["-ccbin", ccbin_path]);
                    }
                    command.arg(p);
                    Some((p, command.spawn()
                        .expect("nvcc failed to start. Ensure that you have CUDA installed and that `nvcc` is in your PATH.").wait_with_output()))
                }
            })
            .collect::<Vec<_>>();

    let ptx_paths: Vec<PathBuf> = glob::glob(&format!("{out_dir}/**/*.ptx"))
        .unwrap()
        .map(|p| p.unwrap())
        .collect();
    // We should rewrite `src/lib.rs` only if there are some newly compiled kernels, or removed
    // some old ones
    let write = !children.is_empty() || kernel_paths.len() < ptx_paths.len();
    for (kernel_path, child) in children {
        let output = child.expect("nvcc failed to run. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        assert!(
            output.status.success(),
            "nvcc error while compiling {kernel_path:?}:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    (write, kernel_paths)
}

#[allow(unused)]
fn compute_cap() -> Result<usize, Error> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let mut compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .expect("Could not parse code")
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=compute_cap")
                .arg("--format=csv")
                .output()
                .expect("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
        let mut lines = out.lines();
        assert_eq!(lines.next().expect("missing line in stdout"), "compute_cap");
        let cap = lines
            .next()
            .expect("missing line in stdout")
            .replace('.', "");
        let cap = cap.parse::<usize>().expect("cannot parse as int {cap}");
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
                .arg("--list-gpu-code")
                .output()
                .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).unwrap();

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().expect("no gpu codes parsed from nvcc");
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute caps
    if !supported_nvcc_codes.contains(&compute_cap) {
        panic!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        panic!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
