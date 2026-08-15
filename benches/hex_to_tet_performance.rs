//! Criterion benchmark for the real hexahedral-to-tetrahedral conversion path.

#![allow(
    missing_docs,
    reason = "criterion_main! generates an undocumented public fn in this crate root"
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gaia::application::hierarchy::hex_to_tet::HexToTetConverter;
use gaia::domain::grid::StructuredHexGridBuilder;

fn bench_hex_to_tet_structured(c: &mut Criterion) {
    let mesh = StructuredHexGridBuilder::new(8, 8, 8).build();
    c.bench_function("hex_to_tet_structured_512_cells", |b| {
        b.iter(|| {
            let converted = HexToTetConverter::convert(black_box(&mesh));
            black_box((converted.cell_count(), converted.face_count()))
        })
    });
}

criterion_group!(benches, bench_hex_to_tet_structured);
criterion_main!(benches);
