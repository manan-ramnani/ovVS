use std::path::PathBuf;

fn main() {
    let dll = std::env::var("IOVS_LIBRARY").unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("build-icpx")
            .join("bin")
            .join("iovs.dll")
            .to_string_lossy()
            .into_owned()
    });
    let lib = iovs::Lib::load(std::path::Path::new(&dll)).expect("load");
    let n: i64 = 20;
    let dim: i64 = 4;
    let mut data = vec![0f32; (n * dim) as usize];
    for i in 0..data.len() {
        data[i] = ((i * 17) % 100) as f32 / 50.0 - 1.0;
    }
    let (labels, dist) = lib.kmeans_predict(&data, n, dim, 2).expect("kmeans");
    let inertia: f32 = dist.iter().sum();
    println!(
        "rust kmeans ok label0={} inertia={:.4} version={}",
        labels[0],
        inertia,
        lib.version()
    );
}
