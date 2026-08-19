use iovs::Lib;
use std::path::PathBuf;

fn main() {
    let path = std::env::var("IOVS_LIBRARY").unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../build/bin/iovs.dll")
            .to_string_lossy()
            .into_owned()
    });
    let lib = Lib::load(PathBuf::from(&path).as_path()).expect("load");
    println!("rust consumer ok version={}", lib.version());
    let a = vec![1.0f32, 0.0, 0.0, 1.0];
    let b = vec![2.0f32, 3.0, 4.0, 5.0];
    let c = lib.gemm(&a, &b, 2, 2, 2).expect("gemm");
    println!("gemm {:?}", c);
}
