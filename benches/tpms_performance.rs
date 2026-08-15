//! Criterion benchmark for the shared TPMS marching-cubes path.

#![allow(
    missing_docs,
    reason = "criterion_main! generates an undocumented public fn in this crate root"
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use gaia::domain::geometry::tpms::{build_tpms_sphere, Gyroid, TpmsParams};

fn bench_gyroid_sphere(c: &mut Criterion) {
    let surface = Gyroid;
    let params = TpmsParams {
        radius: 5.0,
        period: 2.5,
        resolution: 24,
        iso_value: 0.0,
    };

    c.bench_function("tpms_gyroid_sphere_res24", |b| {
        b.iter(|| black_box(build_tpms_sphere(black_box(&surface), black_box(&params))))
    });
}

criterion_group!(benches, bench_gyroid_sphere);
criterion_main!(benches);
