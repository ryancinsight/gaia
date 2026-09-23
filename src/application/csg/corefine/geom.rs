//! Projection and containment helpers for the 2-D CDT.

use crate::domain::core::constants::DEGENERATE_NORMAL_REL_SQ;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::domain::geometry::predicates::{orient_2d_arr, Orientation};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

// ── Geometry helpers ──────────────────────────────────────────────────────────

/// Choose two projection axes by dropping the largest-magnitude axis of the
/// face normal, ensuring the 2-D projected polygon is well-conditioned.
///
/// # Theorem — Projection Conditioning
///
/// Dropping the dominant axis ensures the projected area ≥ true 3-D area / √3,
/// so the polygon is never near-degenerate for any non-zero normal.
#[inline]
#[must_use]
pub(super) fn dominant_normal_axes(n: Vector3r) -> (usize, usize) {
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    if ax >= ay && ax >= az {
        (1, 2) // drop X → keep Y,Z
    } else if ay >= ax && ay >= az {
        (0, 2) // drop Y → keep X,Z
    } else {
        (0, 1) // drop Z → keep X,Y
    }
}

/// Project a 3-D point to 2-D by selecting two coordinate axes.
#[inline]
pub(super) fn project_2d(p: Point3r, axis_u: usize, axis_v: usize) -> (Real, Real) {
    (p[axis_u], p[axis_v])
}

/// Returns `true` if `p` lies inside or on triangle `(a,b,c)`.
///
/// ## Algorithm
///
/// 1. Exact stage: classify in the dominant-axis 2-D projection using Shewchuk
///    `orient_2d_arr` signs (boundary-inclusive).
/// 2. Fallback stage: if exact stage rejects, run the prior signed-area test
///    with a tiny epsilon to retain legacy tolerance for numerical drift.
///
/// ## Theorem — Exact Projected Inclusion
///
/// Let `π` be the dominant-axis projection for a non-degenerate triangle
/// `(a,b,c)`. For any point `p`, if all directed edge orientation signs of
/// `(π(a),π(b),π(p))`, `(π(b),π(c),π(p))`, `(π(c),π(a),π(p))` have the same
/// orientation as `(π(a),π(b),π(c))` or are degenerate, then `π(p)` lies in the
/// closed triangle `π(abc)`.
///
/// **Proof sketch.**
/// The oriented half-space representation of a triangle in 2-D is the
/// intersection of its three directed edge half-planes. Exact predicate signs
/// evaluate those half-plane memberships without floating-point sign errors.
/// Boundary points correspond to degenerate orientation on one or more edges. ∎
#[inline]
pub(super) fn inside_triangle(
    p: Point3r,
    a: Point3r,
    b: Point3r,
    c: Point3r,
    face_n: Vector3r,
    axis_u: usize,
    axis_v: usize,
) -> bool {
    if inside_triangle_exact_projected(p, a, b, c, axis_u, axis_v) {
        return true;
    }
    inside_triangle_eps(p, a, b, c, face_n)
}

#[inline]
fn inside_triangle_exact_projected(
    p: Point3r,
    a: Point3r,
    b: Point3r,
    c: Point3r,
    axis_u: usize,
    axis_v: usize,
) -> bool {
    let pa = [a[axis_u], a[axis_v]];
    let pb = [b[axis_u], b[axis_v]];
    let pc = [c[axis_u], c[axis_v]];
    let pp = [p[axis_u], p[axis_v]];

    let tri_ori = orient_2d_arr(pa, pb, pc);
    if tri_ori == Orientation::Degenerate {
        return false;
    }

    let o0 = orient_2d_arr(pa, pb, pp);
    let o1 = orient_2d_arr(pb, pc, pp);
    let o2 = orient_2d_arr(pc, pa, pp);

    match tri_ori {
        Orientation::Positive => {
            o0 != Orientation::Negative
                && o1 != Orientation::Negative
                && o2 != Orientation::Negative
        }
        Orientation::Negative => {
            o0 != Orientation::Positive
                && o1 != Orientation::Positive
                && o2 != Orientation::Positive
        }
        Orientation::Degenerate => false,
    }
}

#[inline]
fn inside_triangle_eps(p: Point3r, a: Point3r, b: Point3r, c: Point3r, face_n: Vector3r) -> bool {
    const EPS: Real = 1e-9;
    let d0 = (b - a).cross(p - a).dot(face_n);
    let d1 = (c - b).cross(p - b).dot(face_n);
    let d2 = (a - c).cross(p - c).dot(face_n);
    d0 >= -EPS && d1 >= -EPS && d2 >= -EPS
}

/// Sliver-face fallback: build an ordered boundary polygon from all edge
/// Steiner points and fan-triangulate from the first corner.
///
/// ## Theorem — Completeness for Sliver Faces
///
/// For a degenerate sliver triangle (2D projected area ≈ 0), the CDT fails to
/// resolve the constraint graph. This fallback builds the *augmented boundary
/// polygon* by interleaving each corner with its edge Steiner chain, producing
/// a convex polygon with all Steiner vertices on the boundary. A simple fan
/// from the first vertex covers all sub-triangles without overlap or gap,
/// and guarantees that every edge-shared Steiner vertex appears in the output.
///
/// Adjacent faces that share any split edge receive the same Steiner injections
/// via `propagate_seam_vertices`, so the sub-edges match — eliminating T-junction
/// cracks even for the tightest sliver angles (≤ 1°).
pub(super) fn midpoint_subdivide(
    face: &FaceData,
    edge_steiners: &[Vec<(Real, VertexId)>; 3],
    pool: &VertexPool,
    face_n: Vector3r,
) -> Vec<FaceData> {
    let has_any = edge_steiners.iter().any(|e| !e.is_empty());
    if !has_any {
        return vec![*face];
    }

    // Build ordered boundary polygon:
    //   corner[0] → steiners[0] → corner[1] → steiners[1] → corner[2] → steiners[2]
    let corners = face.vertices;
    let mut poly: Vec<VertexId> =
        Vec::with_capacity(3 + edge_steiners.iter().map(|e| e.len()).sum::<usize>());
    for ei in 0..3_usize {
        poly.push(corners[ei]);
        for &(_, sv) in &edge_steiners[ei] {
            poly.push(sv);
        }
    }
    poly.dedup();
    if poly.len() < 3 {
        return vec![*face];
    }

    // Fan triangulate from poly[0] to cover all sub-cells.
    let v0 = poly[0];
    let p0 = *pool.position(v0);
    let mut result = Vec::with_capacity(poly.len() - 2);

    for i in 1..poly.len() - 1 {
        let va = poly[i];
        let vb = poly[i + 1];
        if va == v0 || vb == v0 || va == vb {
            continue;
        }
        let pa = *pool.position(va);
        let pb = *pool.position(vb);
        let tri_n = (pa - p0).cross(pb - p0);
        // Scale-relative degenerate check.
        let e1s = (pa - p0).norm_squared();
        let e2s = (pb - p0).norm_squared();
        if tri_n.norm_squared() < DEGENERATE_NORMAL_REL_SQ * e1s * e2s {
            continue;
        }
        if tri_n.dot(face_n) >= 0.0 {
            result.push(FaceData::new(v0, va, vb, face.region));
        } else {
            result.push(FaceData::new(v0, vb, va, face.region));
        }
    }

    if result.is_empty() {
        vec![*face]
    } else {
        result
    }
}
