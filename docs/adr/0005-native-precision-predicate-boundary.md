# ADR 0005: Native-precision predicate boundary via exact promotion

- Status: Accepted
- Date: 2026-09-23
- Board item: GAIA-002 (branch `arch/gaia-002-native-predicates`)

## Context

The 3-D Bowyer-Watson kernel is generic over `T: Scalar` for storage and
vector arithmetic, but routes every orientation/insphere decision through
`f64`: `point_to_f64_arr` widens each `Point3<T>` to `[f64; 3]` before
`Tetrahedron::new` and `contains_in_circumsphere` call the wrappers in
`domain::geometry::predicates`, which are written against `Real = f64`.

The `Scalar` seam is sealed to `{f32, f64}` and the crate advertises
`IndexedMesh<f32>` for GPU staging, so the 3-D kernel's `f32` instantiation
is part of the public contract. `README.md` § 2 disclosed the gap and
`docs/mesh_library_gap_audit.md` tracked it as a monomorphization finding.

`geometry-predicates` provides Shewchuk adaptive-precision predicates in
`f64` only; reimplementing adaptive expansions at `f32` is a non-goal and the
dependency stays.

## Decision

Parameterize the predicate wrappers over `T: Scalar`: `orient_2d`,
`orient_2d_arr`, `orient_3d`, `orient_3d_pts`, `incircle`, `insphere`, and
the `csg::predicates3d` helpers. Evaluation promotes each coordinate exactly
into the `f64` expansion arithmetic via `NumericElement::to_f64`: lossless
for `f32` (every `f32` value is exactly representable in `f64`), identity for
`f64`. The predicate therefore returns the exact sign of the determinant of
the stored `T`-precision configuration — promotion does not round, so no
decision is made about geometry that is not there, and for `T = f64` the
result is bit-identical to the previous wrappers.

The kernel calls the predicates directly on `Point3<T>` / `[T; 3]`;
`point_to_f64_arr` and the identity helper `fn r` are deleted. The wrappers
monomorphize, and the `f64` promotion is the identity function, so the `f64`
path is unchanged machine code.

Stated limit: decisions for `f32` meshes are exact with respect to the stored
`f32` coordinates; they are not computed in `f32` arithmetic. A naive `f32`
filter would be inexact — exactness is the stronger contract, and lossless
promotion is the mechanism that preserves it.

## Alternatives rejected

- `f64`-only 3-D kernel bound: removes an advertised instantiation
  (`IndexedMesh<f32>` GPU staging), narrows the public contract, and
  forecloses the `Scalar`-seam direction of GAIA-003. The premise for removal
  — a precision the kernel does not honour — fails once promotion is exact.
- `f32`-native Shewchuk reimplementation: adaptive expansions at `f32` are
  not provided by the maintained dependency and reimplementing them is a
  non-goal.
- Keep the `f64` funnel and document it: leaves the seam split and retains
  the exact call-site duplication GAIA-003 must delete.

## Consequences

The predicate seam is generic; every existing `f64` call site compiles
unchanged (`Real = f64`, `T` infers). `IndexedMesh<f32>` tetrahedralization
carries an explicit native-precision contract: storage and non-predicate
arithmetic in `f32`, decisions exact for the stored values. Tests add the
`f32` empty-circumsphere property check (exact assertion; the `f32`
source-to-stored rounding bound is stated at the assertion), a dyadic
differential test proving `f32` and `f64` runs produce identical topology on
`f32`-representable inputs (Delaunay uniqueness under exact predicates), and
predicate-level `f32`/`f64` instantiation equivalence. `README.md` § 2 and
the gap audit close the tracked finding in the same change.

Overturning evidence: a maintained exact-`f32` predicate source would let the
promotion boundary dispatch to it under the same wrapper signatures, with no
call-site churn.
