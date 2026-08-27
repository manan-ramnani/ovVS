use std::path::PathBuf;

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn main() {
    let dll = std::env::var("OVVS_LIBRARY").unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("build")
            .join("bin")
            .join("ovvs.dll")
            .to_string_lossy()
            .into_owned()
    });
    let lib = ovvs::Lib::load(std::path::Path::new(&dll)).expect("load");
    let n: i64 = 12;
    let dim: i64 = 4;
    let k: i64 = 3;
    let mut data = vec![0f32; (n * dim) as usize];
    for i in 0..data.len() {
        data[i] = ((i * 17) % 100) as f32 / 50.0 - 1.0;
    }
    let q = [0.1f32, -0.2, 0.3, 0.0];
    let (nb, _) = lib.brute_search(&data, n, dim, &q, k).expect("search");
    let mut truth: Vec<(f32, i64)> = (0..n)
        .map(|i| {
            let row = &data[(i * dim) as usize..((i + 1) * dim) as usize];
            (l2sq(&q, row), i)
        })
        .collect();
    truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for t in 0..k as usize {
        if nb[t] != truth[t].1 {
            eprintln!("mismatch {} got {} expect {}", t, nb[t], truth[t].1);
            std::process::exit(1);
        }
    }
    println!("rust consumer ok neighbors={:?} version={}", nb, lib.version());
}
