//! 2-D plane basis projection and exact flat-plane detection.
//!
//! ## Algorithm — Exact Coplanar Plane Detection
//!
//! 1. Pick a representative non-degenerate triangle `(a,b,c)` from `faces`.
//! 2. Build a projection basis from `(a,b,c)` (`PlaneBasis::from_triangle`).
//! 3. For every vertex `p` in every face, evaluate `orient3d(a,b,c,p)`.
//! 4. Return `Some(basis)` iff all orientations are exactly `Sign::Zero`.
//!
//! ## Theorem — Coplanarity Equivalence
//!
//! Let `(a,b,c)` be a non-collinear triangle. A point set `P` is coplanar with
//! the plane of `(a,b,c)` iff `orient3d(a,b,c,p) == 0` for every `p ∈ P`.
//!
//! **Proof sketch.**
//! The signed tetrahedral volume determinant `orient3d(a,b,c,p)` is zero
//! exactly when `p` lies in the affine span of `(a,b,c)`. Because `(a,b,c)` is
//! non-collinear, that span is a unique plane. Therefore all-zero determinants
//! are equivalent to global coplanarity. ∎

use crate::application::csg::predicates3d::triangle_is_degenerate_exact;
use crate::domain::core::scalar::{Point3r, Real, Scalar};
use crate::domain::geometry::normal::newell_normal;
use crate::domain::topology::predicates::{orient3d, Sign};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use leto::geometry::{Point3, Vector3};

/// Orthonormal 2-D projection basis for a plane in 3-D.
///
/// Generic over the scalar `T` (defaulting to [`Real`], i.e. `f64`) so the same
/// construction serves the default-precision CSG pipeline and the
/// scalar-generic boundary-loop stitcher.  Naming the bare `PlaneBasis` keeps
/// the default-precision instantiation, so existing callers are unchanged.
pub(crate) struct PlaneBasis<T = Real> {
    pub(crate) origin: Point3<T>,
    pub(crate) u: Vector3<T>,
    pub(crate) v: Vector3<T>,
    pub(crate) normal: Vector3<T>,
}

impl<T: Scalar> PlaneBasis<T> {
    #[expect(
        clippy::many_single_char_names,
        reason = "standard triangle vertex and in-plane basis naming"
    )]
    pub(crate) fn from_triangle(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> Option<Self> {
        let ab = b - a;
        let ac = c - a;
        let n = ab.cross(ac);
        let nl = n.norm();
        if nl < <T as Scalar>::from_f64(1e-20) {
            return None;
        }
        let ul = ab.norm();
        if ul < <T as Scalar>::from_f64(1e-20) {
            return None;
        }
        let u = ab / ul;
        let normal = n / nl;
        let v = {
            let v_raw = normal.cross(u);
            let vl = v_raw.norm();
            if vl < <T as Scalar>::from_f64(1e-20) {
                return None;
            }
            v_raw / vl
        };
        Some(Self {
            origin: *a,
            u,
            v,
            normal,
        })
    }

    /// Build a projection basis spanning the plane of a polygon.
    ///
    /// The plane normal is Newell's method — [`newell_normal`], the crate SSOT —
    /// and the basis origin is the arithmetic centroid of `points`.  `(u, v)` is
    /// an orthonormal in-plane frame obtained by Gram-Schmidt from an axis seed:
    /// exactly the construction the boundary-loop stitcher used to inline at
    /// each of its call sites.  Returns `None` when the polygon has no
    /// well-defined plane (fewer than three points, or exactly degenerate).
    pub(crate) fn from_points_centroid(points: &[Point3<T>]) -> Option<Self> {
        let normal = newell_normal(points)?;

        let one = <T as Scalar>::from_f64(1.0);
        let zero = <T as Scalar>::from_f64(0.0);
        let seed = if normal.x.abs() < <T as Scalar>::from_f64(0.9) {
            Vector3::new(one, zero, zero)
        } else {
            Vector3::new(zero, one, zero)
        };
        let u_raw = seed - normal * seed.dot(normal);
        let ul = u_raw.norm();
        if ul < <T as Scalar>::from_f64(1e-20) {
            return None;
        }
        let u = u_raw / ul;
        let v = normal.cross(u);

        // Centroid origin — arithmetic mean of the loop vertices.
        let inv_n = one / <T as Scalar>::from_usize(points.len());
        let mut sum = Vector3::new(zero, zero, zero);
        for p in points {
            sum += p.coords;
        }
        let origin = Point3::new(sum.x * inv_n, sum.y * inv_n, sum.z * inv_n);

        Some(Self {
            origin,
            u,
            v,
            normal,
        })
    }

    #[inline]
    pub(crate) fn project(&self, p: &Point3<T>) -> [T; 2] {
        let d = p - self.origin;
        [d.dot(self.u), d.dot(self.v)]
    }

    /// Lift a 2-D point (u,v) back to 3-D.
    #[inline]
    pub(crate) fn lift(&self, u: T, v: T) -> Point3<T> {
        self.origin + self.u * u + self.v * v
    }
}

pub(crate) fn detect_flat_plane(faces: &[FaceData], pool: &VertexPool) -> Option<PlaneBasis> {
    let mut basis: Option<PlaneBasis> = None;
    let mut rep_tri: Option<(Point3r, Point3r, Point3r)> = None;
    for face in faces {
        let a = *pool.position(face.vertices[0]);
        let b = *pool.position(face.vertices[1]);
        let c = *pool.position(face.vertices[2]);
        if triangle_is_degenerate_exact(&a, &b, &c) {
            continue;
        }
        if let Some(b0) = PlaneBasis::from_triangle(&a, &b, &c) {
            basis = Some(b0);
            rep_tri = Some((a, b, c));
            break;
        }
    }
    let basis = basis?;
    let (ra, rb, rc) = rep_tri?;
    for face in faces {
        for &vid in &face.vertices {
            if orient3d(&ra, &rb, &rc, pool.position(vid)) != Sign::Zero {
                return None;
            }
        }
    }
    Some(basis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::storage::face_store::FaceData;

    fn p(x: Real, y: Real, z: Real) -> Point3r {
        Point3r::new(x, y, z)
    }

    #[test]
    fn detect_flat_plane_accepts_exactly_coplanar_faces() {
        let mut pool = VertexPool::for_csg();
        let n = Vector3::new(0.0, 0.0, 1.0);

        let v0 = pool.insert_unique(p(0.0, 0.0, 0.0), n);
        let v1 = pool.insert_unique(p(1.0, 0.0, 0.0), n);
        let v2 = pool.insert_unique(p(1.0, 1.0, 0.0), n);
        let v3 = pool.insert_unique(p(0.0, 1.0, 0.0), n);

        let faces = vec![
            FaceData::untagged(v0, v1, v2),
            FaceData::untagged(v0, v2, v3),
        ];
        assert!(detect_flat_plane(&faces, &pool).is_some());
    }

    #[test]
    fn detect_flat_plane_rejects_algebraically_non_coplanar_vertex() {
        let mut pool = VertexPool::for_csg();
        let n = Vector3::new(0.0, 0.0, 1.0);

        let v0 = pool.insert_unique(p(0.0, 0.0, 0.0), n);
        let v1 = pool.insert_unique(p(1.0, 0.0, 0.0), n);
        let v2 = pool.insert_unique(p(0.0, 1.0, 0.0), n);
        let v3 = pool.insert_unique(p(0.25, 0.25, 1.0e-12), n);

        let faces = vec![
            FaceData::untagged(v0, v1, v2),
            FaceData::untagged(v0, v2, v3),
        ];
        assert!(detect_flat_plane(&faces, &pool).is_none());
    }

    #[test]
    fn detect_flat_plane_ignores_degenerate_seed_face() {
        let mut pool = VertexPool::for_csg();
        let n = Vector3::new(0.0, 0.0, 1.0);

        let v0 = pool.insert_unique(p(0.0, 0.0, 0.0), n);
        let v1 = pool.insert_unique(p(1.0, 0.0, 0.0), n);
        let v2 = pool.insert_unique(p(2.0, 0.0, 0.0), n); // collinear with v0,v1
        let v3 = pool.insert_unique(p(0.0, 1.0, 0.0), n);

        let faces = vec![
            FaceData::untagged(v0, v1, v2), // degenerate
            FaceData::untagged(v0, v1, v3), // valid plane seed
        ];
        assert!(detect_flat_plane(&faces, &pool).is_some());
    }

    #[test]
    fn from_points_centroid_spans_planar_loop_with_centroid_origin() {
        let pts = vec![
            p(0.0, 0.0, 0.0),
            p(2.0, 0.0, 0.0),
            p(2.0, 1.0, 0.0),
            p(0.0, 1.0, 0.0),
        ];
        let basis = PlaneBasis::from_points_centroid(&pts).expect("planar loop has a basis");

        // Plane normal is ±Z; the loop is CCW in XY, so Newell gives +Z.
        assert!((basis.normal.z.abs() - 1.0).abs() < 1e-12);
        // Origin is the loop centroid.
        assert!((basis.origin.x - 1.0).abs() < 1e-12);
        assert!((basis.origin.y - 0.5).abs() < 1e-12);
        // Projection is isometric: the first side is 2 units long in the plane.
        let proj: Vec<[Real; 2]> = pts.iter().map(|q| basis.project(q)).collect();
        let side = ((proj[1][0] - proj[0][0]).powi(2) + (proj[1][1] - proj[0][1]).powi(2)).sqrt();
        assert!((side - 2.0).abs() < 1e-10, "expected side 2.0, got {side}");
    }

    #[test]
    fn from_points_centroid_rejects_collinear_loop() {
        let pts = vec![p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(2.0, 0.0, 0.0)];
        assert!(PlaneBasis::from_points_centroid(&pts).is_none());
    }
}
