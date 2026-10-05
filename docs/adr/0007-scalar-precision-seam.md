# ADR 0007: Native-Precision Scalar Seam (`T: Scalar`)

**Status**: Accepted  
**Date**: 2026-10-05  
**Board items**: GAIA-003

## Context

Gaia was initially built on `Real = f64` throughout. Callers working with
single-precision coordinate data (GPU pipelines, embedded sensors) paid two
costs: (1) every quality metric widened `f32` to `f64` before computing, and
(2) the resulting report values carried more apparent precision than the input
warranted.

## Decision

Progressively replace internal `Real` references with a `T: Scalar` type
parameter, keeping `Real` only as the default (`type IndexedMesh = IndexedMesh<f64>`).
Each migration slice:

1. Adds the bound to a module's public and private function signatures.
2. Replaces `0.0`, `1.0`, etc. literals with eunomia constants
   (`NumericElement::ZERO`, `Scalar::from_f64`, `FloatElement::from_count`).
3. Retains a backward-compatible `Real`-typed wrapper where callers rely on
   concrete `f64` types (e.g. the CDT and CSG arrangement backends, which
   require exact-predicate `f64` arithmetic at the correctness boundary).
4. Validates each slice with `cargo test --lib` requiring zero regressions.

## Consequences

**Good:**
- `IndexedMesh<f32>` quality metrics and IO export now execute in native `f32`
  precision without silent widening.
- `HistogramT<T>` and `exact_percentile_scalar<T>` provide native-precision
  statistical summaries.
- `write_binary_stl`, `write_obj`, `write_ply` accept any `IndexedMesh<T>`.
- `TetrahedralQualityCriteria::try_new_typed` / `BoundaryFacetQualityCriteria`
  accept `aequitas` typed quantities preventing unit mix-ups.
- `QualityThresholds::from_typed` uses typed angle and dimensionless quantities.

**Neutral:**
- The CDT / CSG arrangement backend intentionally retains `Real = f64` at the
  exact-predicate boundary (ADR-0005). Generic T passes through to those
  backends by converting coordinates `.to_f64()` at the seam.

**Bad / Risk:**
- Silent conversion from `f32 → f64` at `.to_f64()` boundary is still
  present for CSG callers; correctness is maintained because the predicate
  boundary has always required `f64`.
