# Safety & Validation Reference

Gaia uses **Eunomia** (numeric element abstraction), **Aequitas** (typed
dimensional quantities), and Rust's type system to catch geometry errors at
compile time and validate inputs at runtime before they reach computation.

## Typed Dimensional Thresholds (Aequitas)

`QualityThresholds::from_typed` accepts Aequitas-typed values so
unit mismatches are compile-time errors:

```rust,ignore
use aequitas::{Angle, Dimensionless};
use gaia::quality::QualityThresholds;

let thresholds = QualityThresholds::from_typed(
    Dimensionless::new(50.0),         // max aspect ratio (dimensionless)
    Angle::from_degrees(20.0),        // min face angle in degrees
    Dimensionless::new(0.85),         // max skewness [0, 1]
    Dimensionless::new(5.0),          // max edge ratio
);
```

Passing a length where an angle is expected is a **compile-time error** —
no runtime check required.

## NaN-Propagation Safety

All quality metrics propagate `NaN` explicitly.  If any vertex coordinate is
`NaN`:

- `QualityMetric::from_values` returns a metric with `min = max = mean = NaN`
- `StandardQualityAnalyzer::compute` counts the face as bad for all metrics
- `MeshValidator` reports the mesh as failing

This prevents silent corruption from propagating through CFD pipelines.

## Boundary Outward-Orientation Invariant

For every tetrahedral cell `[v0, v1, v2, v3]` produced by `hex_to_tet`, all
four boundary faces have outward-pointing normals.  The winding theorem is
proved in the [Algorithm Reference](algorithms.md); the invariant is enforced
by the test `hex_to_tet_boundary_faces_are_outward_oriented`.

The check you can apply on any mesh:

```rust,ignore
use gaia::quality::normals::NormalAnalysis;
use aequitas::Dimensionless;

let analysis = NormalAnalysis::compute(mesh)?;
let threshold = Dimensionless::new(0.95); // fraction of consistent normals
assert!(analysis.is_consistent_within(threshold));
```

## Watertightness Checks

Use `WatertightnessReport` to verify a closed mesh has no boundary edges:

```rust,ignore
use gaia::quality::WatertightnessReport;

let report = WatertightnessReport::compute(mesh);
assert!(report.is_watertight(), "mesh has {} boundary edges", report.boundary_edge_count());
```

See the [Watertightness Diagnostics](watertightness.md) chapter for
visualisation of boundary regions.

## Integer Overflow Protection

All mesh dimension arithmetic uses `checked_mul` or Rust's overflow-checking
debug mode.  Key invariants:

- Vertex count ≤ `u32::MAX` (enforced at construction; indices are `u32`)
- Face count ≤ `u32::MAX`
- Grid cell coordinates use `i64` — handles ±9 × 10¹² ε-units without overflow

## Safe IO

All format writers (`STL`, `OBJ`, `PLY`) are generic over `T: Scalar` and
convert to `f32` on the wire only for binary STL (which the format mandates).
No silent widening or narrowing occurs for `f64` meshes written as `f64` ASCII.

```rust,ignore
use gaia::io::{write_stl_binary, write_obj};

// f32 binary STL — exactly what the format stores
write_stl_binary::<f32>(&mesh_f32, &path)?;

// f64 OBJ — vertex coordinates written at full precision
write_obj::<f64>(&mesh_f64, &path)?;
```

## Eunomia Scalar Completeness

`Scalar` is a sealed trait implemented only for `f32` and `f64`.  Downstream
code cannot accidentally instantiate mesh types with `u32`, `bool`, or custom
types that would break geometry invariants.  The sealed design is explained in
`src/domain/core/scalar.rs`.

## Unsafe Code Policy

Gaia contains **zero `unsafe` blocks** in library code.  All geometry
algorithms use safe Rust, with correctness properties verified by the 1149
unit tests and the CI ratchet (`atlas-conformance.py`).
