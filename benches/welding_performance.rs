//! Criterion performance benchmarks for topology-preserving vertex welding.
//!
//! The case that matters is the mesh a CSG boolean or an unwelded import
//! leaves behind: every triangle owns its corner positions, so the position
//! array carries one duplicate per corner and the welder must merge them
//! without ever collapsing a shared-face or non-manifold edge. The grid
//! below builds exactly that shape: `n × n` cells, two triangles per cell,
//! three private position slots per triangle.

#![allow(
    missing_docs,
    reason = "criterion_main! generates an undocumented public fn in this crate root"
)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use gaia::application::welding::MeshWelder;
use gaia::domain::core::index::VertexId;
use gaia::domain::core::scalar::Point3r;
use gaia::domain::core::scalar::Real;
use gaia::domain::core::scalar::Scalar;
use gaia::infrastructure::storage::face_store::{FaceData, FaceStore};

/// One unwelded grid: two triangles per cell, three private positions per
/// triangle, so `positions.len() == cells * 6` with every grid vertex
/// duplicated up to six times (the shared interior corner).
fn unwelded_grid(n: usize) -> (Vec<Point3r>, FaceStore) {
    let mut positions = Vec::with_capacity(n * n * 6);
    let mut faces = FaceStore::with_capacity(n * n * 2);
    let mut push_tri = |a: [Real; 3], b: [Real; 3], c: [Real; 3], positions: &mut Vec<Point3r>| {
        let base = positions.len();
        positions.push(Point3r::new(a[0], a[1], a[2]));
        positions.push(Point3r::new(b[0], b[1], b[2]));
        positions.push(Point3r::new(c[0], c[1], c[2]));
        faces.push(FaceData::untagged(
            VertexId::from_usize(base),
            VertexId::from_usize(base + 1),
            VertexId::from_usize(base + 2),
        ));
    };
    for y in 0..n {
        for x in 0..n {
            let (x0, y0) = (
                <Real as Scalar>::from_usize(x),
                <Real as Scalar>::from_usize(y),
            );
            let (x1, y1) = (
                <Real as Scalar>::from_usize(x + 1),
                <Real as Scalar>::from_usize(y + 1),
            );
            push_tri([x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], &mut positions);
            push_tri([x0, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], &mut positions);
        }
    }
    (positions, faces)
}

fn bench_weld(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertex_weld");
    for n in [64_usize, 128] {
        group.bench_with_input(BenchmarkId::new("unwelded_grid", n), &n, |b, &n| {
            b.iter_with_setup(
                || unwelded_grid(n),
                |(positions, mut faces)| {
                    let result = MeshWelder::new().weld(black_box(&positions), &mut faces);
                    black_box((result.vertices_merged, result.faces_updated))
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_weld);
criterion_main!(benches);
