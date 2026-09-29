//! Adversarial and property-based tests for the CSG arrangement pipeline.
//!
//! These tests target edge cases, degenerate inputs, and scale-regression
//! scenarios that the regular unit tests do not cover.
//!
//! ## Categories
//!
//! | Category | What it tests |
//! |----------|---------------|
//! | Degeneracy | Coaxial tubes, near-parallel faces, coplanar intersections |
//! | Scale | Flat slivers with extreme aspect ratios (millifluidic scale) |
//! | Stability | GWN stability near geometry (near-degenerate inputs) |
//! | Self-intersection | Non-manifold input detection |
//! | Property-based | Proptest invariants: GWN exterior bound, snap determinism |

use crate::application::csg::arrangement::classify::{
    centroid, classify_fragment, tri_normal, FragmentClass,
};
use crate::application::csg::arrangement::gwn::gwn;
use crate::application::csg::boolean::{csg_boolean, BooleanOp};
use crate::application::csg::detect_self_intersect::detect_self_intersections;
use crate::domain::core::constants::{GWN_INSIDE_THRESHOLD, GWN_OUTSIDE_THRESHOLD};
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::primitives::{Cube, Cylinder, PrimitiveMesh};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use proptest::prelude::*;

// ── Helper builders ────────────────────────────────────────────────────

fn unit_cube() -> crate::domain::mesh::IndexedMesh {
    Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("unit_cube build")
}

fn offset_cube(dx: f64) -> crate::domain::mesh::IndexedMesh {
    Cube {
        origin: Point3r::new(-1.0 + dx, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("offset_cube build")
}

/// Build a unit-cube reference mesh for GWN tests.
fn unit_cube_faces() -> (VertexPool, Vec<FaceData>) {
    let mut pool = VertexPool::default_millifluidic();
    let n = leto::geometry::Vector3::zeros();
    let s = 0.5_f64;
    let mut v = |x, y, z| pool.insert_or_weld(Point3r::new(x, y, z), n);
    let c000 = v(-s, -s, -s);
    let c100 = v(s, -s, -s);
    let c010 = v(-s, s, -s);
    let c110 = v(s, s, -s);
    let c001 = v(-s, -s, s);
    let c101 = v(s, -s, s);
    let c011 = v(-s, s, s);
    let c111 = v(s, s, s);
    let f = FaceData::untagged;
    let faces = vec![
        f(c000, c010, c110),
        f(c000, c110, c100),
        f(c001, c101, c111),
        f(c001, c111, c011),
        f(c000, c001, c011),
        f(c000, c011, c010),
        f(c100, c110, c111),
        f(c100, c111, c101),
        f(c000, c100, c101),
        f(c000, c101, c001),
        f(c010, c011, c111),
        f(c010, c111, c110),
    ];
    (pool, faces)
}

// ── Signed-volume helper ─────────────────────────────────────────────

fn signed_volume(mesh: &crate::domain::mesh::IndexedMesh) -> f64 {
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

mod part1;
mod part2;
mod part3;
mod part4;
mod part5;
