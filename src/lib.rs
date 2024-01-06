use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug)]
pub enum Error {}

#[derive(Debug)]
pub struct Builder {
    cuda_root: Option<PathBuf>,
    kernel_paths: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    compute_cap: Option<usize>,
    out_dir: PathBuf,
    extra_args: Vec<&'static str>,
}

impl Default for Builder {
    fn default() -> Self {
        let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
            |_| num_cpus::get_physical(),
            |s| usize::from_str(&s).unwrap(),
        );

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus)
            .build_global()
            .unwrap();
        let out_dir = std::env::var("OUT_DIR").unwrap().into();

        let cuda_root = cuda_include_dir();
        let kernel_paths = default_kernels().unwrap_or(vec![]);
        let include_paths = default_include().unwrap_or(vec![]);
        let extra_args = vec![];
        let compute_cap = compute_cap().ok();
        Self {
            cuda_root,
            kernel_paths,
            include_paths,
            extra_args,
            compute_cap,
            out_dir,
        }
    }
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
    pub fn kernel_paths(mut self, paths: Vec<PathBuf>) -> Self {
        let inexistent_paths: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !inexistent_paths.is_empty(){
            panic!("Kernels paths do not exist {inexistent_paths:?}");
        }
        self.kernel_paths = paths;
        self
    }

    pub fn include_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.include_paths = paths;
        self
    }

    pub fn kernel_paths_glob(mut self, glob: &str) -> Self {
        self.kernel_paths = glob::glob(glob)
            .expect("Invalid blob")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    pub fn include_paths_glob(mut self, glob: &str) -> Self {
        self.include_paths = glob::glob(glob)
            .expect("Invalid blob")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    pub fn out_dir<P: Into<PathBuf>>(mut self, out_dir: P) -> Self {
        self.out_dir = out_dir.into();
        self
    }

    pub fn arg(mut self, arg: &'static str) -> Self {
        self.extra_args.push(arg);
        self
    }

    pub fn cuda_root<P>(&mut self, path: P)
    where
        P: Into<PathBuf>,
    {
        self.cuda_root = Some(path.into());
    }

    pub fn build_lib<P>(self, out_file: P)
    where
        P: Into<PathBuf>,
    {
        let out_file = out_file.into();
        let compute_cap = self.compute_cap.expect("Failed to get compute_cap");
        let out_dir = self.out_dir;
        let cu_files: Vec<_> = self
            .kernel_paths
            .iter()
            .map(|f| {
                let mut obj_file = out_dir.join(f.file_name().unwrap());
                obj_file.set_extension("o");
                (f, obj_file)
            })
            .collect();
        let out_modified: Result<_, _> = out_file.metadata().and_then(|m| m.modified());
        let should_compile = if let Ok(out_modified) = out_modified {
            self.kernel_paths.iter()
                .any(|entry| {
                        let in_modified = entry.metadata().unwrap().modified().unwrap();
                        in_modified.duration_since(out_modified).is_ok()
                })
        } else {
            true
        };
        let ccbin_env = std::env::var("NVCC_CCBIN");
        if should_compile {
            cu_files
            .par_iter()
            .map(|(cu_file, obj_file)| {
                let mut command = std::process::Command::new("nvcc");
                command
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("-c")
                    .args(["-o", obj_file.to_str().unwrap()])
                    .args(["--default-stream", "per-thread"])
                    .args(&self.extra_args);
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = command
                    .spawn()
                    .expect("failed spawning nvcc")
                    .wait_with_output().unwrap();
                if !output.status.success() {
                    panic!(
                        "nvcc error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<(), std::io::Error>>().unwrap();
            let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
            let mut command = std::process::Command::new("nvcc");
            command
                .arg("--lib")
                .args(["-o", out_file.to_str().unwrap()])
                .args(obj_files);
            let output = command
                .spawn()
                .expect("failed spawning nvcc")
                .wait_with_output()
                .unwrap();
            if !output.status.success() {
                panic!(
                    "nvcc error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                    &command,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            }
        }
    }

    pub fn build_ptx(self) -> Result<Bindings, Error> {
        let cuda_root = self.cuda_root.expect("Could not find CUDA in standard locations, set it manually using Builder().set_cuda_root(...)");
        let compute_cap = self.compute_cap.expect("Could not find compute_cap");
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            cuda_root.join("include").display()
        );
        let out_dir = self.out_dir;

        let mut include_directories = self.include_paths;
        for path in &mut include_directories {
            println!("cargo:rerun-if-changed={}", path.display());
            let destination = out_dir.join(path.file_name().unwrap());
            std::fs::copy(path.clone(), destination).unwrap();
            // remove the filename from the path so it's just the directory
            path.pop();
        }

        include_directories.sort();
        include_directories.dedup();

        #[allow(unused)]
        let include_options: Vec<String> = include_directories
            .into_iter()
            .map(|s| "-I".to_string() + &s.into_os_string().into_string().unwrap())
            .collect::<Vec<_>>();

        let ccbin_env = std::env::var("NVCC_CCBIN");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
        let children = self.kernel_paths
            .par_iter()
            .flat_map(|p| {
                println!("cargo:rerun-if-changed={}", p.display());
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
                        .args(["--output-directory", &out_dir.display().to_string()])
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

        let ptx_paths: Vec<PathBuf> = glob::glob(&format!("{0}/**/*.ptx", out_dir.display()))
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

fn cuda_include_dir() -> Option<PathBuf> {
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

    println!("cargo:info={roots:?}");

    #[allow(unused)]
    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    #[cfg(feature = "ci-check")]
    let root: PathBuf = "ci".into();

    #[cfg(not(feature = "ci-check"))]
    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
}

fn compute_cap() -> Result<usize, Error> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
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
