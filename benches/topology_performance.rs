#![allow(clippy::cast_precision_loss)] // bench/test: usize→f64 for ratios
//! Criterion performance benchmarks for the topology repair paths.
//!
//! `orient_outward` and `retain_largest_component` both exist to repair the
//! many-component meshes a CSG boolean leaves behind, so the case that matters
//! is a mesh with *many* components, not a single large one. The benchmarks
//! below scale the component count with the face count held proportional
//! (`4` faces per disjoint tetrahedron), which makes a per-component cost that
//! is linear in the face count show up as quadratic growth in the parameter.
#![allow(
    missing_docs,
    reason = "criterion_main! generates an undocumented public fn in this crate root"
)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use gaia::application::watertight::seal::seal_boundary_loops;
use gaia::domain::core::index::RegionId;
use gaia::domain::core::scalar::Point3r;
use gaia::domain::mesh::IndexedMesh;
use gaia::domain::topology::connectivity::connected_components;
use gaia::domain::topology::AdjacencyGraph;
use gaia::infrastructure::storage::edge_store::EdgeStore;

/// `n` disjoint closed tetrahedra: `n` connected components, `4n` faces.
///
/// Each tetrahedron is placed far enough from its neighbours that no two share
/// a vertex, so the face-adjacency graph has exactly `n` components.
fn disjoint_tetrahedra(n: usize) -> IndexedMesh<f64> {
    tetrahedra(n, 4)
}

/// `n` disjoint tetrahedra keeping their first `faces_per_tet` faces; with
/// three faces each tetrahedron leaves one three-edge hole.
fn tetrahedra(n: usize, faces_per_tet: usize) -> IndexedMesh<f64> {
    let mut mesh: IndexedMesh<f64> = IndexedMesh::with_cell_size(1.0e-6);
    for i in 0..n {
        let o = (i as f64) * 10.0;
        let v0 = mesh.add_vertex_pos(Point3r::new(o, 0.0, 0.0));
        let v1 = mesh.add_vertex_pos(Point3r::new(o + 1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex_pos(Point3r::new(o, 1.0, 0.0));
        let v3 = mesh.add_vertex_pos(Point3r::new(o, 0.0, 1.0));
        for [a, b, c] in [[v0, v1, v2], [v0, v2, v3], [v0, v3, v1], [v1, v3, v2]]
            .into_iter()
            .take(faces_per_tet)
        {
            mesh.add_face(a, b, c);
        }
    }
    mesh
}

/// One connected component of `n` faces: a triangle fan around a shared vertex.
fn single_component(n: usize) -> IndexedMesh<f64> {
    let mut mesh: IndexedMesh<f64> = IndexedMesh::with_cell_size(1.0e-6);
    let hub = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
    let mut rim = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let a = (i as f64) * 0.01;
        rim.push(mesh.add_vertex_pos(Point3r::new(a.cos(), a.sin(), 0.0)));
    }
    for i in 0..n {
        mesh.add_face(hub, rim[i], rim[i + 1]);
    }
    mesh
}

/// Cost of the orientation repair as the component count grows.
fn bench_orient_outward(c: &mut Criterion) {
    let mut group = c.benchmark_group("orient_outward");
    for n in [16usize, 64, 256, 1024] {
        group.bench_with_input(BenchmarkId::new("disjoint", n), &n, |b, &n| {
            b.iter_batched(
                || disjoint_tetrahedra(n),
                |mut mesh| {
                    mesh.orient_outward();
                    black_box(mesh.face_count());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    // Control: the same face counts as one component, so the per-component
    // term is held at 1 and any growth here is the per-face term.
    for n in [64usize, 256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::new("single", n), &n, |b, &n| {
            b.iter_batched(
                || single_component(n),
                |mut mesh| {
                    mesh.orient_outward();
                    black_box(mesh.face_count());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Cost of phantom-island removal as the component count grows.
fn bench_retain_largest(c: &mut Criterion) {
    let mut group = c.benchmark_group("retain_largest_component");
    for n in [16usize, 64, 256, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || disjoint_tetrahedra(n),
                |mut mesh| {
                    black_box(mesh.retain_largest_component());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Cost of the face-adjacency BFS alone, adjacency built outside the timer.
fn bench_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components");
    for n in [64usize, 1024] {
        let mesh = disjoint_tetrahedra(n);
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let adjacency = AdjacencyGraph::build(&mesh.faces, &edges);
        group.bench_with_input(BenchmarkId::new("disjoint", n), &n, |b, _| {
            b.iter(|| black_box(connected_components(&mesh.faces, &adjacency)));
        });
    }
    group.finish();
}

/// Cost of the boundary-loop walk and fan fill: many short holes, then one
/// long rim.
fn bench_seal_boundary_loops(c: &mut Criterion) {
    let mut group = c.benchmark_group("seal_boundary_loops");
    let cases = [
        ("holes", tetrahedra(256, 3)),
        ("holes", tetrahedra(1024, 3)),
        ("rim", single_component(1024)),
    ];
    for (shape, mesh) in cases {
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let n = mesh.face_count();
        group.bench_with_input(BenchmarkId::new(shape, n), &n, |b, _| {
            b.iter_batched(
                || (mesh.vertices.clone(), mesh.faces.clone()),
                |(mut vertices, mut faces)| {
                    black_box(seal_boundary_loops(
                        &mut vertices,
                        &mut faces,
                        &edges,
                        RegionId::INVALID,
                    ))
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_orient_outward,
    bench_retain_largest,
    bench_connected_components,
    bench_seal_boundary_loops
);
criterion_main!(benches);
