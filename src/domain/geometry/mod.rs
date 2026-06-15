//! Geometric primitives, exact predicates, and mesh builders.

pub mod aabb;
pub mod measure;
pub mod normal;
pub mod plane;
pub mod predicates;

pub use aabb::Aabb;
pub use normal::{triangle_area_normal, triangle_centroid, triangle_normal};
pub use plane::Plane;
pub use predicates::Orientation;

pub mod nurbs;
pub use nurbs::{NurbsCurve, NurbsSurface, TessellationOptions};

pub mod primitives;
pub mod tpms;
pub use primitives::{
    Antiprism, BiconcaveDisk, Capsule, Cone, Cube, Cuboctahedron, Cylinder, Disk, Dodecahedron,
    Elbow, Ellipsoid, FischerKochCySphere, FrdSphere, Frustum, GeodesicSphere, GyroidSphere,
    HelixSweep, Icosahedron, IwpSphere, LidinoidSphere, LinearSweep, NeoviusSphere, Octahedron,
    Pipe, Pyramid, RevolutionSweep, RoundedCube, SchwarzDSphere, SchwarzPSphere, SerpentineTube,
    SphericalShell, SplitPSphere, StadiumPrism, Tetrahedron, Torus, TruncatedIcosahedron, UvSphere,
};
