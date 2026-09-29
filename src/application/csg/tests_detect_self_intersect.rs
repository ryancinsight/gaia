//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::Vector3r;
use crate::infrastructure::storage::face_store::FaceData;

fn pool_with_verts(pts: &[[f64; 3]]) -> (VertexPool, Vec<crate::domain::core::index::VertexId>) {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::zeros();
    let ids = pts
        .iter()
        .map(|&[x, y, z]| pool.insert_or_weld(Point3r::new(x, y, z), n))
        .collect();
    (pool, ids)
}

// ── tri_tri_intersects unit tests ─────────────────────────────────────────

/// Two flat XY-plane triangles that overlap → intersects.
#[test]
fn two_triangles_crossing_in_xz_plane_intersect() {
    // ta: flat in XZ, centered at origin.  tb: rotated ~90°, crosses ta.
    let ta = [
        Point3r::new(-1.0, 0.0, 0.0),
        Point3r::new(1.0, 0.0, 0.0),
        Point3r::new(0.0, 0.0, 1.0),
    ];
    let tb = [
        Point3r::new(0.0, -1.0, 0.5),
        Point3r::new(0.0, 1.0, 0.5),
        Point3r::new(0.0, 0.0, -0.5),
    ];
    assert!(
        tri_tri_intersects(&ta, &tb),
        "crossing triangles must be detected"
    );
}

/// Two triangles on opposite sides of a plane → no intersection.
#[test]
fn two_separated_triangles_do_not_intersect() {
    let ta = [
        Point3r::new(0.0, 0.0, 0.0),
        Point3r::new(1.0, 0.0, 0.0),
        Point3r::new(0.0, 1.0, 0.0),
    ];
    // tb translated by +5 on Z → separated.
    let tb = [
        Point3r::new(0.0, 0.0, 5.0),
        Point3r::new(1.0, 0.0, 5.0),
        Point3r::new(0.0, 1.0, 5.0),
    ];
    assert!(
        !tri_tri_intersects(&ta, &tb),
        "separated triangles must not intersect"
    );
}

/// Two coplanar triangles: conservatively returns false.
#[test]
fn coplanar_triangles_return_false() {
    let ta = [
        Point3r::new(0.0, 0.0, 0.0),
        Point3r::new(1.0, 0.0, 0.0),
        Point3r::new(0.0, 1.0, 0.0),
    ];
    let tb = [
        Point3r::new(0.1, 0.1, 0.0),
        Point3r::new(0.5, 0.0, 0.0),
        Point3r::new(0.0, 0.5, 0.0),
    ];
    // Conservative: coplanar → false (not a proper intersection in 3-D).
    assert!(
        !tri_tri_intersects(&ta, &tb),
        "coplanar overlap is not reported"
    );
}

// ── detect_self_intersections integration tests ───────────────────────────

/// A flat quad split into 2 adjacent triangles has no self-intersection.
#[test]
fn adjacent_triangles_are_not_self_intersecting() {
    let (pool, ids) = pool_with_verts(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let faces = vec![
        FaceData::untagged(ids[0], ids[1], ids[2]),
        FaceData::untagged(ids[0], ids[2], ids[3]),
    ];
    let pairs = detect_self_intersections(&faces, &pool);
    assert!(
        pairs.is_empty(),
        "adjacent triangles must not be reported as self-intersecting"
    );
}

/// Two non-adjacent triangles that cross each other ARE detected.
#[test]
fn non_adjacent_crossing_triangles_are_detected() {
    // ta: horizontal XY-plane triangle.
    // tb: tilted to cross ta (shared no vertices).
    let (pool, ids) = pool_with_verts(&[
        // ta
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        // tb (crosses ta through z=0)
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
    ]);
    let faces = vec![
        FaceData::untagged(ids[0], ids[1], ids[2]),
        FaceData::untagged(ids[3], ids[4], ids[5]),
    ];
    let pairs = detect_self_intersections(&faces, &pool);
    assert!(
        !pairs.is_empty(),
        "crossing non-adjacent triangles must be detected"
    );
    assert_eq!(pairs, vec![(0, 1)]);
}

/// Completely separated non-adjacent triangles: empty result.
#[test]
fn separated_non_adjacent_triangles_not_detected() {
    let (pool, ids) = pool_with_verts(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [10.0, 10.0, 10.0],
        [11.0, 10.0, 10.0],
        [10.0, 11.0, 10.0],
    ]);
    let faces = vec![
        FaceData::untagged(ids[0], ids[1], ids[2]),
        FaceData::untagged(ids[3], ids[4], ids[5]),
    ];
    let pairs = detect_self_intersections(&faces, &pool);
    assert!(
        pairs.is_empty(),
        "widely separated triangles must not be reported"
    );
}

/// Single face: always empty result.
#[test]
fn single_face_is_never_self_intersecting() {
    let (pool, ids) = pool_with_verts(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let faces = vec![FaceData::untagged(ids[0], ids[1], ids[2])];
    let pairs = detect_self_intersections(&faces, &pool);
    assert!(pairs.is_empty());
}

/// Adversarial: triangle that "touches" another at a vertex (no proper intersection).
#[test]
fn vertex_touch_is_not_self_intersection() {
    // ta and tc share vertex at (1,0,0) — adjacent by vertex, not edge.
    let (pool, ids) = pool_with_verts(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0], // same position as ids[1]
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    // Since ids[1] and ids[3] are welded to the same position, they have
    // the same VertexId — the adjacency filter must catch this.
    let faces = vec![
        FaceData::untagged(ids[0], ids[1], ids[2]),
        FaceData::untagged(ids[3], ids[4], ids[5]),
    ];
    let pairs = detect_self_intersections(&faces, &pool);
    // Both share vertex ids[1]==ids[3] → adjacency filter removes the pair.
    assert!(
        pairs.is_empty(),
        "vertex-adjacent triangles must not be reported as self-intersecting"
    );
}

/// Triangle in the plane `z = 0`, `scale` on a side.
fn flat_triangle(scale: Real) -> [Point3r; 3] {
    [
        Point3r::new(0.0, 0.0, 0.0),
        Point3r::new(scale, 0.0, 0.0),
        Point3r::new(scale, scale, 0.0),
    ]
}

/// The same triangle tilted about the x axis by `tilt` radians, so its plane
/// meets `z = 0` along the x axis and the two triangles properly cross.
///
/// With `tilt = 1e-12` the normals are `sin²θ ≈ 1e-24` apart: far below the
/// parallel threshold, and shallow enough that an unnormalised parallel test
/// (`|n₁ × n₂|² ∝ L⁸`) crosses its own threshold between scales.
fn crossing_triangle(scale: Real, tilt: Real) -> [Point3r; 3] {
    let s = tilt;
    [
        Point3r::new(0.4 * scale, -0.2 * scale, -0.2 * scale * s),
        Point3r::new(0.6 * scale, 0.2 * scale, 0.2 * scale * s),
        Point3r::new(0.5 * scale, 0.6 * scale, 0.6 * scale * s),
    ]
}

/// The near-parallel decision must not depend on the mesh's scale: this pair
/// is `sin²θ ≈ 1e-24` apart in normal direction, so it is "parallel" at every
/// scale and conservatively reported as non-intersecting.
///
/// Under the previous unnormalised test this failed: `|n₁ × n₂|²` scales as
/// `L⁸`, so at `1e-3` the pair read as parallel and at `1e3` it read as
/// intersecting.
#[test]
fn near_parallel_pair_decisions_are_scale_invariant() {
    let mut decisions = Vec::new();
    for scale in [1e-3_f64, 1.0, 1e3, 1e5] {
        let ta = flat_triangle(scale);
        let tb = crossing_triangle(scale, 1e-12);
        decisions.push(tri_tri_intersects(&ta, &tb));
    }
    assert!(
        decisions.windows(2).all(|w| w[0] == w[1]),
        "the same near-parallel configuration must be decided identically at \
             every scale: {decisions:?}"
    );
    assert!(
        !decisions[0],
        "planes this close to parallel are conservatively non-intersecting"
    );
}

/// A pair whose straddling vertex sits a fixed *fraction* of the mesh scale
/// off the opposing plane must also be decided identically at every scale.
#[test]
fn plane_band_decisions_are_scale_invariant() {
    let mut decisions = Vec::new();
    for scale in [1e-3_f64, 1.0, 1e3, 1e5] {
        let ta = flat_triangle(scale);
        // A second triangle crossing ta's plane, with one vertex a
        // scale-relative 1e-11 above it and another well below.
        let gap = 1e-11 * scale;
        let tb = [
            Point3r::new(0.4 * scale, 0.2 * scale, gap),
            Point3r::new(0.6 * scale, 0.4 * scale, -0.5 * scale),
            Point3r::new(0.5 * scale, 0.8 * scale, 0.5 * scale),
        ];
        decisions.push(tri_tri_intersects(&ta, &tb));
    }
    assert!(
        decisions.windows(2).all(|w| w[0] == w[1]),
        "the plate band is relative to the mesh scale, so this pair must be \
             decided identically everywhere: {decisions:?}"
    );
}
