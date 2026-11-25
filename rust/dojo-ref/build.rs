fn main() {
    let cpu = std::env::var_os("CARGO_FEATURE_CPU").is_some();
    let gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some();
    if cpu && gpu {
        panic!("dojo-ref features `cpu` and `gpu` cannot be enabled together");
    }
    if cpu {
        build_cpu();
    } else {
        build_gpu();
    }
}

fn engine_dir() -> String {
    if let Ok(dir) = std::env::var("CUDA_DOJO_ENGINE_DIR") {
        return dir;
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    format!("{}/../../cuda/engine", manifest)
}

fn build_cpu() {
    let engine = engine_dir();
    for rel in [
        "movegen.cu",
        "make_unmake.cu",
        "attacks.cu",
        "zobrist.cu",
        "cpu/refcpu.cpp",
        "engine_types.h",
        "host_compat.h",
        "movegen.cuh",
        "make_unmake.cuh",
        "attacks.cuh",
        "zobrist.cuh",
        "coord3d.cuh",
    ] {
        println!("cargo:rerun-if-changed={}/{}", engine, rel);
    }

    cc::Build::new()
        .cpp(true)
        .compiler("g++")
        .std("c++17")
        .flag("-Wno-unused-function")
        .define("ENGINE_HOST_ONLY", None)
        .include(&engine)
        // Wrappers are .cpp so the cc crate uses the host C++ compiler.
        // Compiling the .cu files directly can select nvcc.
        .file(format!("{}/cpu/tu_movegen.cpp", engine))
        .file(format!("{}/cpu/tu_make_unmake.cpp", engine))
        .file(format!("{}/cpu/tu_attacks.cpp", engine))
        .file(format!("{}/cpu/tu_zobrist.cpp", engine))
        .file(format!("{}/cpu/refcpu.cpp", engine))
        .compile("refcpu");

    // rustc links with cc, which does not search g++'s private lib directory.
    if let Some(dir) = libstdcxx_dir() {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    println!("cargo:rustc-link-lib=stdc++");
}

fn libstdcxx_dir() -> Option<String> {
    let output = std::process::Command::new("g++")
        .args(["-print-file-name=libstdc++.so"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let path = std::path::Path::new(path.trim());
    path.parent().map(|dir| dir.display().to_string())
}

fn build_gpu() {
    let engine = engine_dir();
    println!("cargo:rerun-if-changed={}/librefcuda.so", engine);
    println!("cargo:rustc-link-search=native={}", engine);
    println!("cargo:rustc-link-lib=dylib=refcuda");

    let cuda_lib = cuda_lib_dir();
    if !std::path::Path::new(&cuda_lib).join("libcudart.so").exists() {
        println!(
            "cargo:warning=libcudart.so not found in {}. Set CUDA_LIB_DIR, or build the CPU rules path with `--no-default-features --features cpu`.",
            cuda_lib
        );
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", engine);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib);
}

fn cuda_lib_dir() -> String {
    if let Ok(dir) = std::env::var("CUDA_LIB_DIR") {
        return dir;
    }
    for candidate in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13.0/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        if std::path::Path::new(candidate).join("libcudart.so").exists() {
            return candidate.to_string();
        }
    }
    "/usr/local/cuda/lib64".to_string()
}
