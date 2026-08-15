# Gaia

Watertight CFD mesh generation and geometry topology kernel for millifluidic devices.

## Installation

The crates.io name `gaia` belongs to an unrelated third-party crate, so this
crate is published as [`gaia-mesh`](https://crates.io/crates/gaia-mesh). The
import path stays `gaia` via the `[lib] name`, so rename the dependency and no
`use gaia::…` changes:

```toml
[dependencies]
gaia = { package = "gaia-mesh", version = "0.4.0" }
```

Default features are empty; enable the I/O and domain features you need:

```toml
[dependencies]
gaia = { package = "gaia-mesh", version = "0.4.0", features = ["stl-io", "vtk-io"] }
```

API documentation is published at [docs.rs/gaia-mesh](https://docs.rs/gaia-mesh).

## Distribution

Publishing runs from the `Crates.io Release` GitHub Actions workflow, triggered
by a published GitHub Release (or manually via `workflow_dispatch` for
validation without publishing). It verifies the packaged source and uses
crates.io Trusted Publishing to obtain a short-lived credential.

The generated [Gaia Mesh Book](https://ryancinsight.github.io/gaia/) is built
from the public mesh builders and published by the `Gaia mesh book` GitHub
Actions workflow after a successful `main` build.

## Core Architecture

Gaia implements exactly-computable geometry and topologically-safe mesh representations to eliminate non-manifold degeneracies typical of floating-point CAD kernels.

### 1. Compile-Time Topological Safety
- **GhostCell + SlotMap Integration**: Mesh entities (vertices, half-edges, faces) are managed via slotmap keys. Mutability is gated by an invariant lifetime brand `'id` using a single `GhostToken<'id>`, guaranteeing that entities cannot be accessed or mutated outside their parent mesh boundaries at compile time (zero runtime overhead).
- **Branded Mesh representation**: Exposes a branded half-edge mesh `Mesh<'id>` for topological traversal, and an `IndexedMesh` for serialized I/O snapshots.

### 2. Numerical Correctness via Robust Predicates
- **Shewchuk Adaptive Precision**: Wraps robust geometric predicates (`orient_2d`, `orient_3d`) for the documented `f64` predicate boundary. The predicate implementation protects sign decisions against roundoff in that representation.
- **Where exactness ends**: exact predicates decide orientation and incircle/insphere signs. They do not decide CSG inside/outside membership — `classify_fragment` thresholds a generalized winding number against the tolerance constants `GWN_INSIDE_THRESHOLD` (0.65) and `GWN_OUTSIDE_THRESHOLD` (0.35) in `domain/core/constants.rs`, and resolves the band between them with coplanarity and nearest-face tiebreakers that are themselves tolerance-based.
- **Precision contract**: Surface and tetrahedral-builder kernels execute native `T` arithmetic; the 3-D Bowyer-Watson kernel currently converts coordinates to `f64` for its robust predicates. Native-precision 3-D predicates remain an audited extension item.

### 3. Validated Volume Construction
- **TetrahedralMeshBuilder**: Builds `IndexedMesh<T>` volume meshes from welded
  vertices and tetrahedral cells, rejects invalid cells before mutation, and
  deduplicates shared triangular faces while preserving outward boundary
  winding. The builder is generic over Gaia's native `f32`/`f64` scalar seam.
- **Structured-grid reuse**: `StructuredGridBuilder` uses the same builder as
  downstream FEM consumers, so face sharing and cell orientation have one
  implementation.
- **Hexahedral source grids**: `StructuredHexGridBuilder` emits triangulated
  boundary faces and hexahedral cells for the `HexToTetConverter` path.
- **Monomorphized numeric kernel**: coordinate access and native-precision
  orientation remain generic over `T`; scalar-independent topology bookkeeping
  is kept in non-generic helpers.

### 4. Allocation-Free & Pre-allocated Subdivisions
- **Midpoint Node Pre-sizing**: `P2MeshConverter` (P1-to-P2 triangle subdivision) pre-allocates target mesh capacity via `empty_clone_with_capacity` and pre-sizes edge midpoint maps.
- **Decomposition Pre-sizing**: `HexToTetConverter` pre-sizes target cell storage and reuses vertex adjacency vectors to avoid dynamic heap allocations during 3D hexahedral decomposition.
- **Fixed-Capacity Hex Decomposition**: hexahedral conversion keeps the eight-vertex scratch set and five/six-tetrahedron selection on the stack, while three-vertex face keys use direct compare-swap canonicalization.
- **Boundary Relaxation**: `SdfMesher` pre-sizes boundary vertex sets and hoists Jacobi relaxation buffers, eliminating per-iteration vector re-allocations.
- **CSG normalization views**: CSG keeps operands inside the stable coordinate
  band borrowed through a static operand view; only scale-normalized operands
  materialize transformed storage.

### 5. Zero-Allocation Seam Propagation
- **Inline Edge Adjacency**: `propagate_seam_vertices_until_stable` builds the undirected edge-to-face adjacency map exactly once per stable convergence loop.
- **AdjacentFaces Struct**: Employs an inline struct (`AdjacentFaces`) holding up to 2 adjacent face indices inline, falling back to a heap-allocated vector only for non-manifold edges. This eliminates dynamic allocations for 99.9% of mesh edges.
- **Hoisted Coordination**: Hoists the crossing parameter buffer `t_params` to reuse its allocation across face loops.

### 6. Memory-Efficient Metadata
- **Cow Boundary Labels**: Stores boundary patch tags as `Cow<'static, str>` in `IndexedMesh::boundary_labels`. This avoids thousands of redundant heap allocations of identical string literals (e.g. `"inlet"`, `"outlet"`, `"wall"`), and reduces cloning cost to a simple pointer copy.

---

## Directory Structure

```text
src/
  lib.rs                     # Public re-exports (IndexedMesh, Mesh<'id>, with_mesh)
  application/               # High-level pipeline and topological operations
    channel/                 # Channel builders (Venturi, Serpentine, Branching)
    csg/                     # Constructive Solid Geometry Boolean ops (Arrangement, Classification)
    delaunay/                # 2D/3D Delaunay triangulation (Bowyer-Watson, Ruppert refinement)
    hierarchy/               # Mesh promotion and decomposition (P2 conversion, Hex-to-Tet)
    pipeline/                # End-to-end mesh generation pipelines
    quality/                 # Surface, boundary-facet, CFD-cell, and tetrahedral quality metrics
    watertight/              # Mesh repair, topological sealing, and manifold verification
    welding/                 # Spatial-hash vertex deduplication and snap-to-grid/vertex
  domain/                    # Domain primitives and invariants
    core/                    # Scalar types, index aliases, and error types
    geometry/                # NURBS curves/surfaces, planes, AABB, and exact predicates
    mesh/                    # IndexedMesh, surface builder, and tetrahedral volume builder
    topology/                # Core cells, element types, and connectivity graphs
  infrastructure/            # I/O formats, spatial indexes, and memory stores
    io/                      # STL, VTK, OpenFOAM, GLTF, OBJ, PLY, and 3MF exporters
    permission/              # GhostCell lifetime brand implementation
    spatial/                 # BVH and SSVDAG spatial indexes
    storage/                 # AttributeStore and VertexPool spatial hash welder
```

---

## Features

- **Constructive Solid Geometry (CSG)**: Watertight Union, Intersection, and
  Difference using BVH broad phase and exact intersection co-refinement. When
  two validated axis-aligned rectangular prisms form one rectangular prism,
  the union reconstructs that closed-form boundary and avoids progressive
  coplanar-subdivision error.
- **NURBS Engine**: B-spline and NURBS boundary representations evaluated via Cox-de Boor recursion with curvature-adaptive surface tessellation.
- **Delaunay Triangulation**: Bowyer-Watson triangulation with Ruppert refinement, using `total_cmp` float sorting for robust NaN handling.
- **Tetrahedral volumes**: `TetrahedralMeshBuilder<T>` validates native-
  precision cells, deduplicates shared faces, and exposes a boundary shell for
  FEM consumers such as Kwavers.
- **Tetrahedral acceptance**: `TetrahedralQualityCriteria<T>` requires
  consumer-supplied radius-edge, dihedral-angle, normalized-volume, and
  optional maximum-volume bounds. It classifies measured cells as accepted,
  sliver, poor-shape, oversized, or invalid without imposing hidden defaults.
- **Boundary acceptance**: `BoundaryFacetQualityCriteria<T>` supplies explicit
  Aequitas-typed angle, dimensionless edge-ratio, and optional SI-length
  edge-size bounds.
  `TetrahedralQualityCriteria<T>::assess_boundary` combines them by face
  incidence and rejects malformed boundary topology instead of treating it as
  interior.
- **CFD I/O Exporters**:
  - **OpenFOAM**: High-performance export with region index mapping.
  - **3MF**: Pre-allocated XML serialization.
  - **Standard formats**: STL, VTK, GLB, OBJ, and PLY.

---

## Testing & Quality Gates

`.github/workflows/ci.yml` runs this sequence on every push and pull request
to `main`. Run it locally first:

```powershell
# Format check
cargo fmt --all -- --check

# Lint floor: clippy::pedantic + clippy::unwrap_used + missing_docs, denied.
# The floor is declared in Cargo.toml [lints]; pre-existing debt is held by a
# counted allow-list there (ratchet GAIA-LINT-1), so a lint class absent from
# that list fails the build.
cargo clippy --all-targets --all-features -- -D warnings

# Tests under the committed .config/nextest.toml budget (30 s slow /
# 60 s terminate). Bare `cargo test` bypasses that instrument.
cargo nextest run --all-features

# Doctests — nextest does not execute them
cargo test --doc --all-features

# Warning-clean rustdoc
$env:RUSTDOCFLAGS = "-D warnings"; cargo doc --no-deps --all-features
```

## License

Dual-licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
