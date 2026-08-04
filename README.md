# Gaia

Watertight CFD mesh generation and geometry topology kernel for millifluidic devices.

## Distribution

Gaia is published as [`gaia`](https://crates.io/crates/gaia). GitHub Releases
tagged `crate-gaia-v<version>` validate the exact package and publish through
the protected `crates-io` environment with a short-lived crates.io Trusted
Publishing credential.

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
  native-precision facet angle, edge-ratio, and optional edge-size bounds.
  `TetrahedralQualityCriteria<T>::assess_boundary` combines them by face
  incidence and rejects malformed boundary topology instead of treating it as
  interior.
- **CFD I/O Exporters**:
  - **OpenFOAM**: High-performance export with region index mapping.
  - **3MF**: Pre-allocated XML serialization.
  - **Standard formats**: STL, VTK, GLB, OBJ, and PLY.

---

## Testing & Quality Gates

Verify compilation correctness, strict lint compliance, and test suite execution:

```powershell
# Format check
cargo fmt --check

# Clippy warnings denial
cargo clippy --all-targets --all-features -- -D warnings

# Execute all tests
cargo nextest run --all-features
```
