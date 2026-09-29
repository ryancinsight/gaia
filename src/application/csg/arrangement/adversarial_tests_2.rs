//! Extended adversarial tests for the CSG arrangement pipeline — Part 2.
//!
//! Covers failure modes that mesh Boolean libraries (Cork, CGAL, libigl,
//! Manifold) are known to struggle with but were not yet covered in
//! `adversarial_tests.rs`.
//!
//! ## Categories
//!
//! | Category | What it tests |
//! |----------|---------------|
//! | Genus > 0 | Torus × Cube — handle non-simply-connected topology |
//! | Mixed orientation | CW + CCW operands — winding robustness |
//! | Many-operand coplanar | ≥ 10 flush cubes via N-ary union |
//! | Sharp dihedral | Near-parallel intersecting planes at 1°–2° angle |
//! | Interior subtraction | A \ B where B is fully interior — cavity topology |
//! | Repeated-scale stability | Union → scale → union → unscale loop |
//! | Near-tangent contact | Cylinders at separation ≈ 2R — grazing topology |
//! | Self-intersection detect | Non-manifold input rejected/detected |

use crate::application::csg::boolean::{csg_boolean, csg_boolean_nary, BooleanOp};
use crate::application::csg::detect_self_intersect::detect_self_intersections;

use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::primitives::{Cube, Cylinder, PrimitiveMesh, Torus, UvSphere};
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

// ── Helper ─────────────────────────────────────────────────────────────

fn signed_volume(mesh: &IndexedMesh) -> f64 {
    let mut vol = 0.0_f64;
    for face in mesh.faces.iter() {
        let a = mesh.vertices.position(face.vertices[0]);
        let b = mesh.vertices.position(face.vertices[1]);
        let c = mesh.vertices.position(face.vertices[2]);
        vol += a.x * (b.y * c.z - b.z * c.y)
            + a.y * (b.z * c.x - b.x * c.z)
            + a.z * (b.x * c.y - b.y * c.x);
    }
    (vol / 6.0).abs()
}

fn unit_cube() -> IndexedMesh {
    Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("unit_cube build")
}

mod part1;
mod part2;
