//! # gaia
//!
//! State-of-the-art watertight CFD mesh generation for millifluidic devices.
//!
//! This crate provides the canonical indexed and half-edge mesh representations,
//! a complete Boolean CSG pipeline, spatial-hash vertex welding, exact
//! geometric predicates, manifold/watertight checking, and OpenFOAM-compatible
//! I/O — all targeting millimetre-scale microfluidic channel geometries.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use gaia::{MeshBuilder, core::scalar::Point3r};
//!
//! let mesh = MeshBuilder::new()
//!     .add_triangle_vertex_positions(/* ... */)
//!     .build();
//! assert!(mesh.is_watertight());
//! ```
//!
//! ## Architecture Diagram
//!
//! ```text
//! ┌─ gaia crate ─────────────────────────────────────────────────────────┐
//! │                                                                           │
//! │  Entry points                                                             │
//! │  ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐   │
//! │  │  with_mesh(f)    │    │   MeshBuilder     │    │  csg_boolean_*   │   │
//! │  │  HalfEdgeMesh    │    │   IndexedMesh     │    │  CsgNode tree    │   │
//! │  └────────┬─────────┘    └────────┬──────────┘    └────────┬─────────┘   │
//! │           │                       │                         │             │
//! │  ┌────────▼─────────┐    ┌────────▼──────────┐   ┌────────▼──────────┐  │
//! │  │ permission/      │    │ storage/           │   │ csg/              │  │
//! │  │  GhostToken      │    │  VertexPool        │   │  BspTree          │  │
//! │  │  GhostCell       │    │  FaceStore         │   │  BvhTree          │  │
//! │  └────────┬─────────┘    │  EdgeStore         │   │  boolean pipeline │  │
//! │           │              └────────┬──────────┘   └───────────────────┘  │
//! │  ┌────────▼─────────┐            │                                       │
//! │  │ topology/        │    ┌────────▼──────────┐                           │
//! │  │  halfedge kernel │    │ geometry/          │                           │
//! │  │  BoundaryPatch   │    │  exact predicates  │                           │
//! │  │  ElementType     │    │  AABB, Plane, NURBS│                           │
//! │  └──────────────────┘    └───────────────────┘                           │
//! │                                                                           │
//! │  Cross-cutting: welding/  watertight/  quality/  io/  core/              │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Module Overview
//!
//! | Module | Contents |
//! |--------|---------|
//! | `domain::mesh` | `HalfEdgeMesh`, `IndexedMesh`, `MeshBuilder` |
//! | `domain::topology` | Half-edge structures, boundary patches, element types |
//! | `domain::geometry` | Exact predicates, AABB, plane, NURBS, builders |
//! | `application::welding` | 26-neighbor `SnappingGrid`, `SpatialHashGrid`, `MeshWelder` |
//! | `infrastructure::storage` | `VertexPool`, `FaceStore`, `EdgeStore`, `SlotPool` |
//! | `application::watertight` | Manifold check, Euler characteristic, repair |
//! | `application::quality` | Triangle quality metrics and validation reports |
//! | `infrastructure::permission` | `GhostToken`, `GhostCell`, `PermissionedArena` |
//! | [`core`] | Scalar types, indices (`VertexKey`, `VertexId`, …), errors |
//! | `infrastructure::io` | STL and VTK mesh I/O |
//! | `application::csg` | BSP-tree + BVH boolean operations |
//!
//! ## Invariants
//!
//! The following mesh invariants are enforced at all API boundaries:
//!
//! 1. **Manifold half-edge**: every interior edge is shared by exactly 2 faces;
//!    `twin(twin(he)) == he` and `next(prev(he)) == he`.
//! 2. **Spatial deduplication**: `VertexPool` and `SnappingGrid` guarantee that
//!    no two vertex positions are closer than `TOLERANCE` apart.
//! 3. **Watertight closure**: `IndexedMesh::is_watertight()` verifies zero
//!    boundary edges (every edge has exactly 2 adjacent faces).
//! 4. **Generational key safety**: `VertexKey` / `FaceKey` values are valid
//!    only within the mesh that created them; stale keys return `None`.

// The lint floor lives in Cargo.toml `[lints]`, where CI can deny it. The
// block that used to sit here declared `warn` levels nothing ever ran under
// `-D warnings`, so it enforced nothing; twelve of its allows were also dead
// (the lints no longer fire, or no longer exist in the pedantic group).
//
// Test code is exempt from the panic-policy lint: an `unwrap()` in a test is
// the assertion, not an input-dependent failure path. This is the crate-root
// half of that carve-out; the example, bench, bin and integration-test targets
// carry their own, since they are separate crates.
#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "test code: an unwrap is the assertion, not an input-dependent failure path"
    )
)]

pub mod application;
pub mod domain;
pub mod infrastructure;

/// Unified error type for all gaia operations.
pub use domain::core::error::{Error, ErrorContext, NurbsKind, Result};

/// Canonical watertight-first indexed surface mesh.
pub use domain::mesh::IndexedMesh;

/// Ergonomic builder for `IndexedMesh`.
pub use domain::mesh::MeshBuilder;

/// Validated tetrahedral volume-mesh builder.
pub use domain::mesh::TetrahedralMeshBuilder;

/// Branded half-edge surface mesh.
pub use domain::mesh::HalfEdgeMesh;

/// Canonical entry point for branded mesh operations.
pub use domain::mesh::with_mesh;

// ── Convenience re-exports ────────────────────────────────────────────────────

/// Normal-orientation analysis report for `IndexedMesh` surfaces.
pub use application::quality::{analyze_normals, NormalAnalysis};

/// Named CFD boundary patch (Inlet / Outlet / Wall / Symmetry / Periodic).
pub use domain::topology::halfedge::BoundaryPatch;

/// CFD boundary patch type discriminant.
pub use domain::topology::halfedge::PatchType;

/// Open polyline — the canonical curve type for tractography streamlines.
pub use domain::geometry::Polyline;

/// Polyline construction and operation errors.
pub use domain::geometry::PolylineError;

/// Deterministic geodesic samples carrying a unit-vector invariant.
pub use domain::geometry::UnitSphereDirectionSet;

/// Exact Shewchuk orientation result.
pub use domain::geometry::Orientation;

/// Directed half-line with a normalized direction and its construction error.
pub use domain::geometry::{Ray, RayError};

/// Axis-aligned bounding box — the ray/voxel traversal companion to [`Ray`]
/// (`Ray::intersect_aabb`), consumed by downstream imaging and dose ray-tracers.
pub use domain::geometry::Aabb;

/// Analytic mesh primitives (26 builders from tetrahedron to truncated icosahedron).
pub use domain::geometry::primitives;

/// Primitive builder re-exports for ergonomic top-level access.
pub use domain::geometry::{
    Antiprism,
    BiconcaveDisk,
    Capsule,
    Cone,
    Cube,
    Cuboctahedron,
    Cylinder,
    Disk,
    Dodecahedron,
    Elbow,
    Ellipsoid,
    // TPMS expansion: Neovius, Lidinoid, I-WP, Split P, FRD, Fischer-Koch C(Y)
    FischerKochCySphere,
    FrdSphere,
    Frustum,
    GeodesicSphere,
    GyroidSphere,
    HelixSweep,
    Icosahedron,
    IwpSphere,
    LidinoidSphere,
    LinearSweep,
    NeoviusSphere,
    Octahedron,
    Pipe,
    Pyramid,
    RevolutionSweep,
    RoundedCube,
    SchwarzDSphere,
    SchwarzPSphere,
    SerpentineTube,
    SphericalShell,
    SplitPSphere,
    StadiumPrism,
    Tetrahedron,
    Torus,
    TruncatedIcosahedron,
    UvSphere,
};

/// Application-level channel builders.
pub use application::channel::{
    BranchingMeshBuilder, ChannelPath, ChannelPathError, ChannelProfile, SerpentineMeshBuilder,
    SubstrateBuilder, SweepError, SweepMesher, VenturiMeshBuilder,
};
