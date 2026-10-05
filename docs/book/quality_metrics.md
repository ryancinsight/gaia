# Quality Metrics Reference

Gaia provides a layered set of mesh quality metrics for surface validation,
tetrahedral volume assessment, and CFD boundary-layer checking. All metric
functions are generic over `T: Scalar` — they execute in the input precision
(`f32` or `f64`) without silent widening.

> This book is built and hosted at **<https://ryancinsight.github.io/gaia/>**
> via the `Gaia mesh book` GitHub Actions workflow.

---

## Surface Triangle Quality

### Triangle Quality Score

```rust,ignore
use gaia::application::quality::triangle::{
    aspect_ratio, min_angle, edge_length_ratio_native, triangle_angles,
};

let a = mesh.vertices.position(face.vertices[0]);
let b = mesh.vertices.position(face.vertices[1]);
let c = mesh.vertices.position(face.vertices[2]);

let ar   = aspect_ratio(a, b, c);   // 1.0 = equilateral, larger = worse
let ma   = min_angle(a, b, c);      // minimum interior angle (radians)
let elr  = edge_length_ratio_native(a, b, c); // shortest/longest ∈ (0,1]
```

### Full Quality Report

The `StandardQualityAnalyzer` produces per-metric histograms and aggregate
counts in one pass. `QualityThresholds::from_typed` accepts aequitas typed
quantities to prevent unit mix-ups:

```rust,ignore
use gaia::application::quality::analyzer::{
    QualityAnalyzer, StandardQualityAnalyzer,
};
use gaia::application::quality::validation::QualityThresholds;
use aequitas::systems::si::quantities::{Angle, Dimensionless};

// Typed thresholds: angle in radians, ratios dimensionless
let analyzer = StandardQualityAnalyzer {
    thresholds: QualityThresholds::from_typed(
        Dimensionless::from_base(10.0),          // max aspect ratio
        Angle::from_base(15_f64.to_radians()),   // min angle
        Dimensionless::from_base(0.85),          // max skewness
        Dimensionless::from_base(0.05),          // min edge ratio
    ),
    n_histogram_bins: 20,
};

let report = analyzer.compute(&mesh);
println!("Bad angle count:  {}", report.bad_angle_count);
println!("Bad aspect count: {}", report.bad_aspect_count);
println!("Fraction failing angle: {:.1}%",
         report.bad_angle_fraction() * 100.0);

if let Some(hist) = &report.min_angle_histogram {
    println!("Angle histogram ({} bins, range [{:.1}°, {:.1}°]):",
             hist.n_bins(),
             hist.min.to_degrees(),
             hist.max.to_degrees());
    for i in 0..hist.n_bins() {
        let mid = hist.midpoint(i).to_degrees();
        println!("  [{:.1}°]: {}", mid, hist.bins[i]);
    }
}
```

### Native-Precision Histograms

`HistogramT<T>` computes histograms in the input scalar precision without
widening to `f64`. Use it on `f32` meshes to stay in single precision:

```rust,ignore
use gaia::application::quality::{vertex_mean_curvature, histograms::HistogramT};

// f32 mesh — all histogram arithmetic stays in f32
let curvatures_f32: Vec<f32> = vertex_mean_curvature(&mesh_f32);
if let Some(hist) = HistogramT::compute(&curvatures_f32, 15) {
    println!("Curvature range: [{:.4}, {:.4}]", hist.min, hist.max);
    println!("Mean bin width: {:.4}", hist.bin_width());
    // Exact percentile in f32 precision
    use gaia::application::quality::histograms::exact_percentile_scalar;
    if let Some(p50) = exact_percentile_scalar::<f32>(&curvatures_f32, 0.5) {
        println!("Median curvature (f32): {:.4}", p50);
    }
}
```

---

## Tetrahedral Volume Quality

### Per-Cell Metrics

```rust,ignore
use gaia::application::quality::{
    tetrahedron_quality, TetrahedronQuality,
};

if let Some(q) = tetrahedron_quality(cell_points) {
    println!("Volume:            {:.6}", q.volume);
    println!("Radius-edge ratio: {:.4}", q.radius_edge_ratio);
    println!("Min dihedral:      {:.1}°", q.min_dihedral_angle.to_degrees());
    println!("Normalized volume: {:.4}", q.normalized_volume);
}
```

### Typed Acceptance Criteria (Aequitas)

The `TetrahedralQualityCriteria` constructors accept `aequitas` dimensioned
quantities to prevent unit confusion:

```rust,ignore
use gaia::application::quality::{
    TetrahedralQualityCriteria, TetrahedralQualityAcceptance,
};
use aequitas::systems::si::quantities::{Angle, Dimensionless};

let criteria = TetrahedralQualityCriteria::<f64>::try_new_typed(
    Dimensionless::from_base(2.0),     // max radius-edge ratio (dimensionless)
    Angle::from_base(0.35),            // min dihedral angle (radians ≈ 20°)
    Dimensionless::from_base(0.25),    // min normalized volume (dimensionless)
    None,
)
.expect("criteria within domain");

if let Some(acceptance) = criteria.assess(&volume_mesh) {
    let total = acceptance.accepted_cell_count
        + acceptance.sliver_count
        + acceptance.poor_shape_count
        + acceptance.oversized_cell_count
        + acceptance.invalid_cell_count;
    println!("Accepted: {}/{} ({:.1}%)",
             acceptance.accepted_cell_count, total,
             100.0 * acceptance.accepted_cell_count as f64 / total as f64);
    println!("Slivers:  {}", acceptance.sliver_count);
    println!("Passed:   {}", acceptance.passed());
}
```

### Quality Report with Statistics

```rust,ignore
use gaia::application::quality::tetrahedral_quality_report;

if let Some(report) = tetrahedral_quality_report(&mesh) {
    for (name, metric) in [
        ("Volume",       &report.volume),
        ("Radius-edge",  &report.radius_edge_ratio),
        ("Dihedral (°)", &report.min_dihedral_angle),
        ("Norm. volume", &report.normalized_volume),
    ] {
        if let Some(m) = metric {
            println!("{name:15}: min={:.4}  mean={:.4}  max={:.4}",
                     m.min, m.mean, m.max);
        }
    }
    println!("Valid cells:   {}", report.valid_cell_count);
    println!("Invalid cells: {}", report.invalid_cell_count);
}
```

---

## Boundary Facet Quality

Boundary-facet criteria use aequitas `Angle`, `Dimensionless`, and `Length`
types so callers cannot accidentally pass a skewness ratio as an angle:

```rust,ignore
use gaia::application::quality::{
    BoundaryFacetQualityCriteria, TetrahedralQualityCriteria,
};
use aequitas::systems::si::quantities::{Angle, Dimensionless, Length};

let cell_criteria = TetrahedralQualityCriteria::<f64>::try_new_typed(
    Dimensionless::from_base(2.0),
    Angle::from_base(0.35),
    Dimensionless::from_base(0.25),
    None,
).expect("cell criteria valid");

let facet_criteria = BoundaryFacetQualityCriteria::<f64>::try_new(
    Angle::from_base(0.5),            // min interior angle (≈ 28.6°)
    Dimensionless::from_base(0.4),    // min edge-length ratio
    Some(Length::from_base(5e-3)),    // max edge length (5 mm)
).expect("facet criteria valid");

if let Some(acceptance) = cell_criteria.assess_boundary(&mesh, &facet_criteria) {
    println!("Boundary cells:    {}", acceptance.boundary_cell_count);
    println!("Accepted boundary: {}", acceptance.accepted_boundary_cell_count);
    let facets = &acceptance.boundary_facet_acceptance;
    println!("Facets accepted:   {}", facets.accepted_facet_count);
    println!("Facets poor shape: {}", facets.poor_shape_facet_count);
    println!("Facets oversized:  {}", facets.oversized_facet_count);
    println!("All passed:        {}", acceptance.passed());
}
```

---

## CFD Volume-Cell Quality (OpenFOAM-compatible)

Non-orthogonality and skewness are the two primary `checkMesh` metrics for
CFD solvers. Gaia computes them for internal faces of volume meshes:

```rust,ignore
use gaia::application::quality::cell_quality::{
    cell_quality_report, face_non_orthogonality, face_skewness,
};

// Per-face metrics
let non_orth = face_non_orthogonality(face_id, owner_cell, neighbour_cell, &mesh);
let skewness = face_skewness(face_id, owner_cell, neighbour_cell, &mesh);

// Full mesh report
if let Some(report) = cell_quality_report(&mesh) {
    let no = &report.non_orthogonality;
    println!("Non-orthogonality: min={:.1}°  mean={:.1}°  max={:.1}°",
             no.min, no.mean, no.max);
    println!("High non-orth (>70°): {}/{} faces",
             report.high_non_orthogonality_count, report.internal_face_count);
    let sk = &report.skewness;
    println!("Skewness: min={:.3}  mean={:.3}  max={:.3}",
             sk.min, sk.mean, sk.max);
    println!("High skewness (>0.85): {}/{} faces",
             report.high_skewness_count, report.internal_face_count);
}
```

### OpenFOAM threshold guidance

| Metric            | Good    | Acceptable | Invalid  |
|-------------------|---------|------------|----------|
| Non-orthogonality | < 70°   | < 85°      | ≥ 90°    |
| Skewness          | < 0.50  | < 0.85     | ≥ 1.0    |

---

## Vertex Mean Curvature

Mean curvature at each vertex via the cotangent-weighted Laplace-Beltrami
operator. Generic over `T: Scalar`:

```rust,ignore
use gaia::application::quality::vertex_mean_curvature;

// f64 mesh
let curvatures: Vec<f64> = vertex_mean_curvature(&mesh);
let finite_count = curvatures.iter().filter(|v| v.is_finite()).count();
let max_h = curvatures.iter()
    .copied()
    .filter(|v| v.is_finite())
    .fold(0.0_f64, f64::max);
println!("Finite curvature sites: {}/{}", finite_count, curvatures.len());
println!("Maximum mean curvature: {:.4}", max_h);
```

For a tessellated sphere of radius `R`, the discrete mean curvature converges
to `1/R` as O(h²) in the edge length `h`. Non-finite values indicate
degenerate local geometry (zero-area faces in the 1-ring).

---

## Surface Normal Analysis

```rust,ignore
use gaia::application::quality::{analyze_normals, NormalAnalysis};

let analysis: NormalAnalysis = analyze_normals(&mesh);
println!("All normals finite: {}", analysis.all_finite);
println!("All normals unit:   {}", analysis.all_unit);
```

---

## Watertightness and Topology

See the [watertightness chapter](watertightness.md) for the full diagnostic
including boundary-edge and Euler-characteristic checks.
