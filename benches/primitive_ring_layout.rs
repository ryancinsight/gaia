//! Criterion benchmark for the swept primitives' cross-section ring table
//! (ATLAS-ARCH-008).
//!
//! The tube and shell primitives built their rings as one `Vec` per station.
//! The conversion keeps the rings in one contiguous buffer with a fixed
//! stride, so a sweep costs one allocation instead of one per station, and
//! the wall builder reads adjacent rings from adjacent memory.
//!
//! The bench drives the public primitive builders, so the same file measures
//! both revisions: the pre-conversion tree (rings as `Vec<Vec<_>>`) and this
//! one (the flat ring table). Run it with
//! `cargo bench -p gaia-mesh --bench primitive_ring_layout`.

#![allow(
    missing_docs,
    reason = "criterion_main! generates an undocumented public fn in this crate root"
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use gaia::domain::geometry::primitives::{HelixSweep, PrimitiveMesh, SerpentineTube};

fn bench_serpentine_tube(c: &mut Criterion) {
    let tube = SerpentineTube {
        tube_radius: 0.3,
        bend_radius: 1.5,
        straight_length: 8.0,
        n_passes: 4,
        tube_segments: 32,
        straight_segments: 12,
        bend_segments: 24,
    };

    c.bench_function("serpentine_tube_rings", |b| {
        b.iter(|| black_box(tube.build().expect("serpentine tube builds")));
    });
}

fn bench_helix_sweep(c: &mut Criterion) {
    let helix = HelixSweep {
        coil_radius: 3.0,
        tube_radius: 0.5,
        pitch: 2.0,
        turns: 3.0,
        tube_segments: 32,
        arc_segments_per_turn: 48,
    };

    c.bench_function("helix_sweep_rings", |b| {
        b.iter(|| black_box(helix.build().expect("helix sweep builds")));
    });
}

criterion_group!(benches, bench_serpentine_tube, bench_helix_sweep);
criterion_main!(benches);
