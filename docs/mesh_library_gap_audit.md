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

## Capability comparison

| Capability | Gaia baseline | Reference-library expectation | Disposition |
| --- | --- | --- | --- |
| Indexed surface topology | Welded indexed triangles, persistent edge adjacency, boundary labels, half-edge view | Common indexed/half-edge or compact mesh representations | Present; retain Gaia ownership |
| Surface validation | Boundary/non-manifold edge counts, winding consistency, signed volume, Euler diagnostic, BVH self-intersection detector | Robust topology and geometric-defect validation | Fixed in this audit: outward orientation is now required and Euler genus is diagnostic. Self-intersection detection remains a separate opt-in operation |
| Surface CSG | BVH broad phase, exact orientation predicates, co-refinement, coplanar arrangement, n-ary operation | Robust Boolean/CSG trees and self-intersection resolution | Present for Gaia's supported triangle-surface contract; add unresolved seam cases to regression data before broadening claims |
| 2-D constrained meshing | Delaunay, PSLG, CDT, Ruppert refinement, metric and smoothing paths | Constrained Delaunay refinement with quality/size criteria | Present; expand property and adversarial coverage |
| 3-D unconstrained meshing | Bowyer-Watson tetrahedralization and deterministic SDF/BCC seeding | Delaunay-based volume meshing | Present as a seed/tetrahedralization kernel; new direct regression coverage added in this audit |
| 3-D constrained refinement | No boundary-feature protection, sizing field, radius-edge refinement loop, or sliver optimization API | CGAL Mesh_3 and TetGen expose constrained/refined quality meshing | Open P1 capability gap; this is the primary state-of-the-art volume-meshing extension |
| Tetrahedral quality | Native `T` volume, radius-edge, minimum-dihedral, and normalized-volume metrics; explicit consumer-supplied acceptance criteria; CFD internal-face metrics | Radius-edge, dihedral-angle, sliver, volume, and size criteria | Present for cell-level acceptance; boundary-feature acceptance and refinement remain open |
| Remeshing/repair | Degenerate-face removal, orientation repair, boundary stitching/sealing, component retention, self-intersection detection | Self-intersection remeshing, decimation, isotropic/adaptive remeshing | Open P1: detection exists, canonical validation does not reject/report it, and no remeshing result is owned by Gaia |
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

## Remaining prioritized work

1. Add boundary-cell quality acceptance rules on top of the native tetrahedral
   criteria. The current policy is cell-level and does not yet encode feature
   protection or surface-facet criteria.
2. Add a constrained 3-D refinement stage around the existing tetrahedralizer:
   protected boundary features, a validated sizing field, explicit termination
   limits, and a quality-improvement phase. The API should be introduced only
   after an Atlas consumer contract and a benchmark workload exist.
3. Promote self-intersection status into an explicit validation policy or
   result type. Do not silently fold the existing detector into every hot
   watertight check until its scalar boundary, coplanar policy, and cost are
   specified.
4. Replace the current f64-only robust-predicate boundary in the 3-D
   tetrahedralizer with an explicit predicate-precision contract. The current
   implementation is generic in `T` but routes orientation/insphere decisions
   through f64 predicates; this is a monomorphization audit finding, not a
   native-precision claim.
5. Add controlled benchmarks and memory measurements for the refinement,
   quality, self-intersection, and SDF paths. Source-level preallocation is
   not evidence of lower RSS or faster execution.

These open items are intentionally not represented as completed capability.
The corresponding implementation/test work belongs in dependency order:
predicate contract and tetra quality before constrained refinement; explicit
self-intersection policy before repair/remeshing; matched benchmarks before
parallel or layout claims.
