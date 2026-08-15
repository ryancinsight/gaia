# Mesh-library gap audit

Date: 2026-08-04

This audit compares Gaia's current mesh-provider surface with the capabilities
that are explicit in the primary documentation for CGAL Mesh_3, Gmsh, TetGen,
and libigl. It is a capability audit, not a claim that every feature of those
libraries belongs in Gaia. Gaia remains the Atlas single source of truth for
the mesh representation and provider-owned algorithms.

## Baseline

The baseline is the Gaia tree at the start of this audit. The public mesh
surface is represented by `IndexedMesh<T>`, `MeshBuilder<T>`,
`TetrahedralMeshBuilder<T>`, structured grids, CSG, Delaunay/CDT, SDF volume
meshing, channel builders, quality analysis, watertight checks/repair, and the
format exporters. The reviewed book gallery enumerates the public mesh-family
boundary; it is intentionally exhaustive by family, not by every continuous
parameter value.

## Atlas consumer audit

The local Atlas consumers confirm Gaia's ownership boundary and identify the
next contracts without requiring consumer-owned mesh algorithms:

- `CFDrs/crates/cfd-schematic-mesh` consumes the Gaia package as `cfd-mesh`.
  Its direct 3-D workload is the existing `BowyerWatson3D` and `SdfMesher`
  path, with a committed `CFDrs/benches/delaunay_3d.rs` benchmark. The
  schematic-to-mesh bridge remains in CFDrs because it depends on CFD
  schematics; Gaia owns the general mesh and tetrahedral kernels.
- `kwavers/crates/kwavers-mesh/src/tetrahedral/gaia.rs` treats
  `gaia::IndexedMesh<f64>` as the authoritative volume artifact. Its current
  contract requires tetrahedral cells, four unique vertex references, finite
  coordinates, adjacency, and Gaia boundary labels before conversion into the
  solver-facing representation. Kwavers retains that solver representation to
  avoid a provider-consumer cycle.
- RITK consumes Gaia `IndexedMesh<f64>` and `MeshBuilder` for surface
  filtering, VTK/STL/OBJ/PLY/GLB I/O, and watertight surface interchange. No
  current RITK contract requires constrained 3-D volume refinement.
- Helios consumes Gaia `Aabb` and `Ray` geometry; its current contract does
  not require a mesh generator.
- Kwavers still reports a missing grid-to-tetrahedral generator at
  `kwavers-simulation/src/solver_factory.rs`. This is a valid future Gaia
  provider driver, but its acceptance contract must cover structured-grid
  ownership, boundary labels, cell orientation, and solver conversion before
  a public refinement API is added.

This audit therefore keeps Gaia as the SSOT for mesh representation and
general algorithms, while leaving domain-specific bridges in their consumers.
The Aequitas-typed boundary criteria introduced in this slice provide the
first typed physical-dimension contract at the Gaia quality boundary; the
remaining consumer bridges still expose their existing `f64` solver/API
contracts and are not silently widened here.

## Capability comparison

| Capability | Gaia baseline | Reference-library expectation | Disposition |
| --- | --- | --- | --- |
| Indexed surface topology | Welded indexed triangles, persistent edge adjacency, boundary labels, half-edge view | Common indexed/half-edge or compact mesh representations | Present; retain Gaia ownership |
| Surface validation | Boundary/non-manifold edge counts, winding consistency, signed volume, Euler diagnostic, BVH self-intersection detector, explicit opt-in status/error | Robust topology and geometric-defect validation | Fixed in this audit: outward orientation is required, Euler genus is diagnostic, and self-intersection rejection is explicit without changing the hot default path |
| Surface CSG | BVH broad phase, exact orientation predicates, co-refinement, coplanar arrangement, n-ary operation | Robust Boolean/CSG trees and self-intersection resolution | Present for Gaia's supported triangle-surface contract; add unresolved seam cases to regression data before broadening claims |
| 2-D constrained meshing | Delaunay, PSLG, CDT, Ruppert refinement, metric and smoothing paths | Constrained Delaunay refinement with quality/size criteria | Present; expand property and adversarial coverage |
| 3-D unconstrained meshing | Bowyer-Watson tetrahedralization and deterministic SDF/BCC seeding | Delaunay-based volume meshing | Present as a seed/tetrahedralization kernel; new direct regression coverage added in this audit |
| 3-D constrained refinement | No boundary-feature protection, sizing field, radius-edge refinement loop, or sliver optimization API | CGAL Mesh_3 and TetGen expose constrained/refined quality meshing | Open P1 capability gap; this is the primary state-of-the-art volume-meshing extension |
| Tetrahedral quality | Native `T` volume, radius-edge, minimum-dihedral, and normalized-volume metrics; explicit cell and boundary-facet acceptance criteria; CFD internal-face metrics | Radius-edge, dihedral-angle, sliver, volume, facet, and size criteria | Present for acceptance; feature protection and constrained refinement remain open |
| Remeshing/repair | Degenerate-face removal, orientation repair, boundary stitching/sealing, component retention, self-intersection detection | Self-intersection remeshing, decimation, isotropic/adaptive remeshing | Open P1: detection is explicit but remains opt-in; a policy-bearing validation result and Gaia-owned remeshing result are not yet defined |
| High-order/mixed volume cells | Linear tetrahedral builder, hexahedral source grid, P2 surface refinement | Mixed/high-order element workflows vary by library | Open P2 and consumer-driven; do not add public cell variants without an Atlas consumer contract |
| Parallel meshing | Parallel CSG classification paths; deterministic serial Delaunay/SDF topology | Gmsh exposes threaded 3-D generation; HXT provides parallel Delaunay/refinement research | Open P2 performance track; benchmark before changing topology or order |
| Interoperability | STL, VTK, OpenFOAM, OBJ, PLY, GLB, 3MF paths | Broad import/export and downstream solver integration | Present for current Atlas consumers; add contract tests as new consumers land |

The comparison is grounded in the primary references: CGAL describes
Delaunay refinement driven by surface and cell criteria followed by quality
optimization; Gmsh documents Delaunay, Frontal-Delaunay, MeshAdapt, size
constraints, and thread controls; TetGen documents constrained Delaunay
tetrahedralization, quality generation, refinement, adaption, and coarsening;
libigl documents exact-arithmetic CSG composition, self-intersection
remeshing, and remeshing/decimation operations.

- [CGAL 3D Mesh Generation](https://doc.cgal.org/latest/Mesh_3/index.html)
- [Gmsh reference manual](https://gmsh.info/doc/texinfo/)
- [TetGen user manual](https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual.html)
- [libigl tutorial](https://libigl.github.io/tutorial/)
- [libigl self-intersection remeshing API](https://libigl.github.io/dox/remesh__self__intersections_8h.html)

## Corrective slice delivered

### Quality precision and invalid geometry

`MeshValidator::validate<T>` previously converted every vertex to `f64`
before computing aspect ratio, angle, skewness, and edge ratio. The public
generic seam therefore did not provide native-precision execution. The
triangle kernels and reductions now operate in `T`; conversion to `f64` is
limited to the report boundary. Non-finite metrics from degenerate faces are
explicit failures instead of passing through NaN comparisons.

Regression coverage includes a native `f32` validation instantiation and a
degenerate face that must fail the quality report.

### Watertight orientation and genus

`check_watertight` now requires a finite positive signed volume in addition to
closed manifold topology and consistent winding. A globally inverted closed
mesh is therefore rejected. `assert_watertight` no longer rejects a valid
closed manifold solely because its Euler characteristic differs from the
genus-zero reference; a torus is a valid watertight mesh with characteristic
zero. The Euler value remains diagnostic metadata.

Regression coverage includes the inverted cube and a torus through the
fallible assertion API.

### 3-D Delaunay coverage

The Bowyer-Watson 3-D kernel had no direct regression module. Tests now cover
retention of a non-degenerate input point set, index validity, distinct
tetrahedron vertices, non-zero native volumes, and rejection of a point
outside a tetrahedron's circumsphere.

### Tetrahedral quality coverage

The quality module now compiles and exports the previously orphaned CFD
internal-face metrics. It also reports native-precision tetrahedral volume,
radius-edge ratio, minimum interior dihedral angle, and equilateral-normalized
volume. Invalid tetrahedral cells are counted without defaulting their
metrics. Tests cover analytical `f32` and `f64` instantiations, a sliver, an
invalid cell, and the structured-grid CFD report.

### Explicit tetrahedral acceptance criteria

`TetrahedralQualityCriteria<T>` now validates consumer-supplied shape bounds
and an optional maximum cell volume in native scalar precision. Assessment is
one traversal over tetrahedral cells and classifies radius-edge-safe cells
with simultaneously low dihedral angle and normalized volume as sliver
candidates. Other shape failures, oversized cells, and malformed cells stay
distinct, so a consumer can select the corrective action without a hidden
default threshold. Scale/translation invariance and `f32`/`f64` assessment
coverage are regression-tested.

### Boundary-facet and boundary-cell acceptance

`BoundaryFacetQualityCriteria<T>` now provides explicit native-precision
Aequitas-typed angle, dimensionless shortest-to-longest edge-ratio, and
optional SI-length edge-size bounds. `TetrahedralQualityCriteria<T>::assess_boundary` identifies exposed
facets by exactly-one-cell incidence, measures each facet once, and accepts a
boundary cell only when its volume-cell policy and every exposed facet policy
pass. Malformed facet references, invalid vertex identifiers, degenerate
facets, and malformed tetrahedral cell topology are counted as invalid rather
than silently classified as interior.

This closes the acceptance-oracle gap, not the refinement gap. It does not
claim feature protection, a sizing field, constrained Delaunay refinement, or
sliver optimization.

### Explicit self-intersection policy

The watertight API now keeps self-intersection detection opt-in. The default
`check_watertight` path reports `SelfIntersectionStatus::NotChecked` and does
not build the BVH. `check_watertight_with_self_intersections` performs the
existing BVH/narrow-phase scan without copying the face store, reports either
`Clear` or the crossing-pair count, and folds a found crossing into the
watertight result. The corresponding assertion API returns the existing typed
`MeshError::SelfIntersection` for the first pair.

The predicate contract remains deliberately narrow: shared-vertex/edge
adjacency is excluded, coplanar overlap is not classified as a proper 3-D
crossing, and the detector currently uses Gaia's `f64` CSG predicate boundary.
Those are explicit limits, not hidden validation defaults.

## Remaining prioritized work

1. Add a constrained 3-D refinement stage around the existing tetrahedralizer:
   protected boundary features, a validated sizing field, explicit termination
   limits, and a quality-improvement phase. The API should be introduced only
   after an Atlas consumer contract and a benchmark workload exist.
2. Extend the self-intersection predicate contract to classify coplanar
   overlap, touching, and near-degenerate cases for callers that need a full
   geometric-defect policy. The current opt-in crossing policy intentionally
   remains limited to proper non-adjacent 3-D intersections.
3. Replace the current f64-only robust-predicate boundary in the 3-D
   tetrahedralizer with an explicit predicate-precision contract. The current
   implementation is generic in `T` but routes orientation/insphere decisions
   through f64 predicates; this is a monomorphization audit finding, not a
   native-precision claim.
4. Add controlled benchmarks and memory measurements for the refinement,
   quality, self-intersection, and SDF paths. Source-level preallocation is
   not evidence of lower RSS or faster execution.

These open items are intentionally not represented as completed capability.
The corresponding implementation/test work belongs in dependency order:
predicate contract and tetra quality before constrained refinement; explicit
self-intersection policy before repair/remeshing; matched benchmarks before
parallel or layout claims.
