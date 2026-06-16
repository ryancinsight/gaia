# Gaia

Watertight CFD mesh generation and geometry topology kernel for millifluidic devices.

## Core Architecture

Gaia implements exactly-computable geometry and topologically-safe mesh representations to eliminate non-manifold degeneracies typical of floating-point CAD kernels.

### 1. Compile-Time Topological Safety
- **GhostCell + SlotMap Integration**: Mesh entities (vertices, half-edges, faces) are managed via slotmap keys. Mutability is gated by an invariant lifetime brand `'id` using a single `GhostToken<'id>`, guaranteeing that entities cannot be accessed or mutated outside their parent mesh boundaries at compile time (zero runtime overhead).
- **Branded Mesh representation**: Exposes a branded half-edge mesh `Mesh<'id>` for topological traversal, and an `IndexedMesh` for serialized I/O snapshots.

### 2. Numerical Correctness via Exact Predicates
- **Shewchuk Adaptive Precision**: Wraps robust geometric predicates (`orient_2d`, `orient_3d`) to perform exact orientation and incircle tests. All topological sign decisions are immune to floating-point roundoff or cancellation.
- **Evidence Tier**: Mathematical verification via Shewchuk's adaptive precision algorithms; validated empirically through full-suite regression tests.

### 3. Allocation-Free & Pre-allocated Subdivisions
- **Midpoint Node Pre-sizing**: `P2MeshConverter` (P1-to-P2 triangle subdivision) pre-allocates target mesh capacity via `empty_clone_with_capacity` and pre-sizes edge midpoint maps.
- **Decomposition Pre-sizing**: `HexToTetConverter` pre-sizes target cell storage and reuses vertex adjacency vectors to avoid dynamic heap allocations during 3D hexahedral decomposition.
- **Boundary Relaxation**: `SdfMesher` pre-sizes boundary vertex sets and hoists Jacobi relaxation buffers, eliminating per-iteration vector re-allocations.

### 4. Zero-Allocation Seam Propagation
- **Inline Edge Adjacency**: `propagate_seam_vertices_until_stable` builds the undirected edge-to-face adjacency map exactly once per stable convergence loop.
- **AdjacentFaces Struct**: Employs an inline struct (`AdjacentFaces`) holding up to 2 adjacent face indices inline, falling back to a heap-allocated vector only for non-manifold edges. This eliminates dynamic allocations for 99.9% of mesh edges.
- **Hoisted Coordination**: Hoists the crossing parameter buffer `t_params` to reuse its allocation across face loops.

### 5. Memory-Efficient Metadata
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
    quality/                 # Aspect ratio, skewness, and normal quality metrics
    watertight/              # Mesh repair, topological sealing, and manifold verification
  domain/                    # Domain primitives and invariants
    core/                    # Scalar types, index aliases, and error types
    geometry/                # NURBS curves/surfaces, planes, AABB, and exact predicates
    mesh/                    # IndexedMesh and Half-Edge mesh models
    topology/                # Core cells, element types, and connectivity graphs
  infrastructure/            # I/O formats, spatial indexes, and memory stores
    io/                      # STL, VTK, OpenFOAM, GLTF, OBJ, PLY, and 3MF exporters
    permission/              # GhostCell lifetime brand implementation
    spatial/                 # BVH and SSVDAG spatial indexes
    storage/                 # AttributeStore and VertexPool spatial hash welder
```

---

## Features

- **Constructive Solid Geometry (CSG)**: Watertight Union, Intersection, and Difference using BVH broad phase and exact intersection co-refinement.
- **NURBS Engine**: B-spline and NURBS boundary representations evaluated via Cox-de Boor recursion with curvature-adaptive surface tessellation.
- **Delaunay Triangulation**: Bowyer-Watson triangulation with Ruppert refinement, using `total_cmp` float sorting for robust NaN handling.
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
