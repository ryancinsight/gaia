# Performance & Precision Guide

Gaia provides two floating-point precisions — `f64` (default) and `f32` —
controlled by a zero-cost generic parameter `T: Scalar`.  Choosing the right
precision, understanding where boundaries occur, and knowing which operations
are performance-sensitive allows you to get the best throughput without
sacrificing correctness.

## Choosing a Precision

| Precision | Tolerance | Typical use |
|-----------|-----------|-------------|
| `f64` | 1 nm (`1e-9`) | High-fidelity CFD, validation, OpenFOAM export |
| `f32` | 10 µm (`1e-5`) | GPU-side geometry staging, memory-bandwidth-limited pipelines |

Both coexist in the same binary — no recompilation or feature flag needed:

```rust,ignore
use gaia::{IndexedMesh, Sphere};

// f64 mesh for simulation
let hi: IndexedMesh<f64> = Sphere { radius: 0.01, ..Default::default() }.build()?;

// f32 mesh for GPU upload
let lo: IndexedMesh<f32> = Sphere { radius: 0.01_f32, ..Default::default() }.build()?;
```

## The Scalar Precision Seam (ADR-0007)

Some algorithms internally require `f64` regardless of the mesh's `T`:

| Algorithm | Fixed precision | Reason |
|-----------|----------------|--------|
| CDT (Constrained Delaunay Triangulation) | `f64` | Shewchuk exact-arithmetic predicates |
| GWN (Generalised Winding Number) | `f64` | Numerical stability of solid-angle accumulation |
| `SnappingGrid::from_point` | `T` | Hash quantisation preserves mesh precision |

The seam is documented in `docs/adr/0007-scalar-precision-seam.md`.  Inputs
are converted to `f64` at the seam boundary and results converted back to
`T`.  No mesh data leaks across — the conversion is explicit and local.

## Memory Layout

`IndexedMesh<T>` stores:

- **Vertices**: `Vec<T>` flat storage — 3 scalars per vertex, packed
- **Faces**: `Vec<FaceIndices>` — compact `u32` indices, 3 per face
- **Normals**: `Option<Vec<T>>` — 3 scalars per vertex, computed on demand

For 1 M vertices at `f32`, the vertex buffer is **12 MiB**; at `f64` it is
**24 MiB**.  For scenes where geometry fits in GPU L2 cache, `f32` halves
bandwidth and typically doubles throughput.

## Weld Tolerance

`SnappingGrid` uses `T::tolerance()` by default:

```rust,ignore
use gaia::welding::SnappingGrid;
use gaia::domain::core::scalar::Scalar;

// Custom 1 µm tolerance at f64
let grid: SnappingGrid<f64> = SnappingGrid::with_epsilon(1e-6);
```

Tighter tolerances reduce false merges on fine geometry; looser tolerances
reduce vertex counts on coarse import.

## Build Performance

| Operation | Typical cost | Notes |
|-----------|-------------|-------|
| `Sphere` (6 rings, 12 segments) | < 1 µs | Pre-computed closed-form |
| CDT triangulation (1000 vertices) | ~100 µs | Shewchuk O(n log n) |
| Vertex welding (1 M verts) | ~50 ms | Hash-grid O(n) expected |
| Normal computation (1 M faces) | ~20 ms | SIMD-friendly, embarrassingly parallel |
| STL export (1 M tris) | ~30 ms | Direct raw write, no buffering overhead |

Benchmarks run on a single core.  Vertex welding and normal computation
scale linearly with thread count.

## Allocation Patterns

Gaia avoids hidden allocations inside hot loops.  The main alloc points are:

1. **Mesh construction** — `Vec::with_capacity(n)` sized from geometry formulas
2. **CDT** — transient PSLG allocation, freed immediately after triangulation
3. **IO export** — single write buffer, streamed to disk

Use the `mnemosyne` allocator integration for profiling allocation hot spots
in long-running CFD preprocessing pipelines.

## LTO and Inlining

All Gaia arithmetic is `#[inline(always)]` on `Scalar` trait methods, so
cross-crate LTO (link-time optimisation) propagates scalar specialisations
through the call graph.  Add to your release profile:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

This typically yields a 15–25% throughput improvement on geometry-heavy
workloads.
