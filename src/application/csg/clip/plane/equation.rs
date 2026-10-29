//! Cached plane equations and face/plane classification types.

use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::geometry::predicates::{orient_3d, Orientation};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

// ── PlaneEquation ─────────────────────────────────────────────────────────────

/// A cached plane equation for efficient batch classification and clipping.
///
/// Stores both the three defining points (for exact `orient_3d` calls) and the
/// float normal + offset (for fast intersection-point computation).
///
/// # Theorem — Caching Preserves Exactness
///
/// `orient_3d(pa, pb, pc, q)` depends only on the coordinates of `pa, pb, pc,
/// q`.  Storing `[pa, pb, pc]` as `[[f64; 3]; 3]` and passing them to
/// `orient_3d` produces bit-identical results to recomputing from a
/// `VertexPool` on every call.  ∎
#[derive(Clone, Debug)]
pub struct PlaneEquation {
    /// Three CCW-oriented points defining the plane (exact `orient_3d` input).
    pa: [f64; 3],
    pb: [f64; 3],
    pc: [f64; 3],
    /// Float normal vector (unnormalised cross product `(pb-pa) × (pc-pa)`).
    normal: [f64; 3],
    /// `normal · pa` — the float plane offset.
    offset: f64,
}

impl PlaneEquation {
    /// Construct from three CCW-ordered points.
    #[must_use]
    #[expect(
        clippy::similar_names,
        reason = "standard point-A/B/C coordinate-array naming for plane construction"
    )]
    pub fn from_points(pa: &Point3r, pb: &Point3r, pc: &Point3r) -> Self {
        let point_a_coords = [pa.x, pa.y, pa.z];
        let point_b_coords = [pb.x, pb.y, pb.z];
        let point_c_coords = [pc.x, pc.y, pc.z];

        let ab = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
        let ac = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let offset = n[0] * pa.x + n[1] * pa.y + n[2] * pa.z;

        Self {
            pa: point_a_coords,
            pb: point_b_coords,
            pc: point_c_coords,
            normal: n,
            offset,
        }
    }

    /// Construct from a [`FaceData`] and [`VertexPool`].
    #[must_use]
    pub fn from_face(face: &FaceData, pool: &VertexPool) -> Self {
        let pa = pool.position(face.vertices[0]);
        let pb = pool.position(face.vertices[1]);
        let pc = pool.position(face.vertices[2]);
        Self::from_points(pa, pb, pc)
    }

    /// Exact orientation classification of a point against this plane.
    ///
    /// Uses Shewchuk `orient_3d` — no floating-point sign error.
    #[inline]
    #[must_use]
    pub fn classify(&self, point: &Point3r) -> Orientation {
        orient_3d(self.pa, self.pb, self.pc, [point.x, point.y, point.z])
    }

    /// Float signed distance (positive = same side as normal).
    ///
    /// Uses the unnormalised normal, so the magnitude is proportional to
    /// distance × ‖normal‖.  Only the *sign* is reliable for topology; use
    /// [`classify`](Self::classify) for exact decisions.
    #[inline]
    #[must_use]
    pub fn signed_distance_unnorm(&self, point: &Point3r) -> f64 {
        self.normal[0] * point.x + self.normal[1] * point.y + self.normal[2] * point.z - self.offset
    }

    /// Compute the plane–edge intersection along `s → e`.
    ///
    /// Returns the interpolated 3-D point.  The parameter `t = ds / (ds - de)`
    /// where `ds, de` are the (unnormalised) signed distances of `s, e`.
    #[must_use]
    pub fn intersect_edge(&self, s: &Point3r, e: &Point3r) -> Point3r {
        let ds = self.signed_distance_unnorm(s);
        let de = self.signed_distance_unnorm(e);
        let denom = ds - de;
        if denom.abs() < 1e-30 {
            return *s;
        }
        let t = ds / denom;
        Point3r::new(
            s.x + (e.x - s.x) * t,
            s.y + (e.y - s.y) * t,
            s.z + (e.z - s.z) * t,
        )
    }

    /// The unnormalised normal vector as `[f64; 3]`.
    #[inline]
    #[must_use]
    pub fn normal_array(&self) -> [f64; 3] {
        self.normal
    }

    /// The unnormalised normal as a `Vector3r`.
    #[inline]
    #[must_use]
    pub fn normal_vec(&self) -> Vector3r {
        Vector3r::new(self.normal[0], self.normal[1], self.normal[2])
    }
}

// ── Face classification ───────────────────────────────────────────────────────

/// Classification of a face's vertices against a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacePlaneClass {
    /// All vertices are positive or degenerate — fully inside the half-space.
    Inside,
    /// All vertices are negative or degenerate — fully outside the half-space.
    Outside,
    /// At least one positive and one negative — face must be split.
    Straddling,
    /// All vertices are exactly on the plane.
    Coplanar,
}
