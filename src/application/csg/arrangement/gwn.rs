//! Generalized Winding Number (GWN) computation for CSG fragment classification.
//!
//! ## Theorem — GWN Correctness
//!
//! For a closed orientable 2-manifold M and a query point q not on M:
//!
//! ```text
//! GWN(q, M) = (1/4π) Σ Ω(q, tri_i)
//! ```
//!
//! where Ω(q, T) is the solid angle subtended by triangle T at q, computed via
//! the van Oosterom–Strackee (1983) formula:
//!
//! ```text
//! Ω = 2·atan2( a·(b×c), |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b| )
//! ```
//!
//! where a, b, c are vectors from q to each triangle vertex (not normalised).
//! **Interior**: GWN = ±1  **Exterior**: GWN = 0
//!
//! ## Theorem — Guard Correctness (Near-vertex skip)
//!
//! `T::min_positive_value()` is the smallest positive normal float for type T:
//! - f64: ≈ 2.2 × 10⁻³⁰⁸  (safe for all physical geometry)
//! - f32: ≈ 1.2 × 10⁻³⁸   (safe for f32 meshes; old 1e-40 cast was 0.0 in f32)
//!
//! Any `norm_squared()` below this threshold means the query is within
//! sub-ULP distance of a mesh vertex — geometrically impossible for any
//! physical model.  Skipping such faces prevents `atan2(0, 0) → NaN`. ∎
//!
//! ## References
//!
//! - van Oosterom & Strackee (1983), *The Solid Angle of a Plane Triangle*,
//!   IEEE Trans. Biomed. Eng. 30(2).
//! - Jacobson et al. (2013), *Robust Inside-Outside Segmentation using
//!   Generalized Winding Numbers*, ACM SIGGRAPH.

use crate::domain::core::constants::{GWN_DENOMINATOR_GUARD, GWN_SOLID_ANGLE_CLIP};
use crate::domain::core::scalar::{Point3r, Scalar, Vector3r};
use crate::domain::geometry::normal::triangle_centroid;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

// ── PreparedFace ──────────────────────────────────────────────────────────────

/// Prepared immutable triangle data for repeated fragment classification.
///
/// Stores copied positions plus derived centroid and unnormalised face normal.
/// This removes repeated `VertexPool` lookups and vector recomputation when
/// classifying many fragments against the same reference mesh.
///
/// ## Theorem — Classification Equivalence
///
/// Each field is copied exactly from the pool without arithmetic transformation.
/// All predicates (van Oosterom solid-angle, orient3d, signed-distance) produce
/// identical results whether reading from `PreparedFace` or querying the pool
/// on demand. ∎
#[derive(Copy, Clone, Debug)]
pub struct PreparedFace {
    pub(crate) a: Point3r,
    pub(crate) b: Point3r,
    pub(crate) c: Point3r,
    pub(crate) centroid: Point3r,
    pub(crate) normal: Vector3r,
    /// Triangle area = ‖normal‖ / 2 (precomputed to avoid redundant cross
    /// products in BVH construction and area-based skip criteria).
    pub(crate) area: f64,
}

/// Build prepared reference-face geometry for repeated classification queries.
#[must_use]
pub fn prepare_classification_faces(
    faces: &[FaceData],
    pool: &VertexPool<f64>,
) -> Vec<PreparedFace> {
    #[cfg(feature = "parallel")]
    {
        use moirai::ParallelSlice;
        faces.par().map_collect(|face| {
            let a = *pool.position(face.vertices[0]);
            let b = *pool.position(face.vertices[1]);
            let c = *pool.position(face.vertices[2]);
            let ab = b - a;
            let ac = c - a;
            let normal = ab.cross(ac);
            let area = 0.5 * normal.norm();
            let centroid = triangle_centroid::<f64>(&a, &b, &c);
            PreparedFace {
                a,
                b,
                c,
                centroid,
                normal,
                area,
            }
        })
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut prepared = Vec::with_capacity(faces.len());
        for face in faces {
            let a = *pool.position(face.vertices[0]);
            let b = *pool.position(face.vertices[1]);
            let c = *pool.position(face.vertices[2]);
            let ab = b - a;
            let ac = c - a;
            let normal = ab.cross(ac);
            let area = 0.5 * normal.norm();
            let centroid = triangle_centroid::<f64>(&a, &b, &c);
            prepared.push(PreparedFace {
                a,
                b,
                c,
                centroid,
                normal,
                area,
            });
        }
        prepared
    }
}

// ── GWN computation ───────────────────────────────────────────────────────────

/// Generalized Winding Number (GWN) of `query` with respect to a closed
/// triangle mesh.
///
/// Returns a value in `[-1, 1]`. For a consistently oriented, watertight
/// surface away from its boundary, the ideal values are ±1 inside and 0
/// outside. A query on a triangulated boundary has no universal pointwise
/// value; direct evaluation may select a side-dependent value.
///
/// ## Implementation Note — Norm Efficiency
///
/// Vector norms (|a|, |b|, |c|) appear only in the denominator and are
/// computed exactly once per vertex.  The near-vertex guard uses
/// `norm_squared() < ε²` (no sqrt) for efficiency.
pub fn gwn<T: Scalar>(
    query: &leto::geometry::Point3<T>,
    faces: &[FaceData],
    pool: &VertexPool<T>,
) -> T {
    let mut solid_angle_sum = <T as Scalar>::from_f64(0.0);
    let near_sq = <T as Scalar>::from_f64(f64::MIN_POSITIVE);
    let one_e_30 = <T as Scalar>::from_f64(GWN_DENOMINATOR_GUARD);
    let two = <T as Scalar>::from_f64(2.0);
    let four_pi = <T as Scalar>::from_f64(4.0 * std::f64::consts::PI);

    for face in faces {
        let a = pool.position(face.vertices[0]);
        let b = pool.position(face.vertices[1]);
        let c = pool.position(face.vertices[2]);

        let va = leto::geometry::Vector3::new(a.x - query.x, a.y - query.y, a.z - query.z);
        let vb = leto::geometry::Vector3::new(b.x - query.x, b.y - query.y, b.z - query.z);
        let vc = leto::geometry::Vector3::new(c.x - query.x, c.y - query.y, c.z - query.z);

        if va.norm_squared() < near_sq || vb.norm_squared() < near_sq || vc.norm_squared() < near_sq
        {
            continue;
        }

        let la = va.norm();
        let lb = vb.norm();
        let lc = vc.norm();

        let num = va.dot(vb.cross(vc));
        let den = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;

        if (den).abs() > one_e_30 || (num).abs() > one_e_30 {
            solid_angle_sum += two * (num).atan2(den);
        }
    }
    (solid_angle_sum / four_pi).clamp(<T as Scalar>::from_f64(-1.0), <T as Scalar>::from_f64(1.0))
}

#[inline(always)]
fn solid_angle_f64(
    va: leto::geometry::Vector3<f64>,
    vb: leto::geometry::Vector3<f64>,
    vc: leto::geometry::Vector3<f64>,
) -> f64 {
    let la = va.norm();
    let lb = vb.norm();
    let lc = vc.norm();
    let num = va.dot(vb.cross(vc));
    let den = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;
    if den.abs() > GWN_DENOMINATOR_GUARD || num.abs() > GWN_DENOMINATOR_GUARD {
        2.0 * num.atan2(den)
    } else {
        0.0
    }
}

/// Compute vertex offsets from query to each PreparedFace vertex.
///
/// Returns `None` if the query lies within sub-ULP distance of any vertex
/// (near-vertex guard — prevents `atan2(0, 0) → NaN`).
#[inline(always)]
pub(super) fn vertex_offsets(
    query: &Point3r,
    face: &PreparedFace,
) -> Option<(
    leto::geometry::Vector3<f64>,
    leto::geometry::Vector3<f64>,
    leto::geometry::Vector3<f64>,
)> {
    let va =
        leto::geometry::Vector3::new(face.a.x - query.x, face.a.y - query.y, face.a.z - query.z);
    let vb =
        leto::geometry::Vector3::new(face.b.x - query.x, face.b.y - query.y, face.b.z - query.z);
    let vc =
        leto::geometry::Vector3::new(face.c.x - query.x, face.c.y - query.y, face.c.z - query.z);
    if va.norm_squared() < f64::MIN_POSITIVE
        || vb.norm_squared() < f64::MIN_POSITIVE
        || vc.norm_squared() < f64::MIN_POSITIVE
    {
        return None;
    }
    Some((va, vb, vc))
}

/// GWN against precomputed `PreparedFace` geometry (f64-only hot path).
///
/// Semantically equivalent to `gwn::<f64>` but avoids pool lookups.
/// Declared `pub(crate)` since only the `arrangement` module needs it.
#[inline]
pub(crate) fn gwn_prepared(query: &Point3r, faces: &[PreparedFace]) -> f64 {
    let mut solid_angle_sum = 0.0_f64;
    for face in faces {
        if let Some((va, vb, vc)) = vertex_offsets(query, face) {
            solid_angle_sum += solid_angle_f64(va, vb, vc);
        }
    }
    (solid_angle_sum / (4.0 * std::f64::consts::PI)).clamp(-1.0, 1.0)
}

/// Bounded GWN against precomputed `PreparedFace` geometry (f64-only).
///
/// Each triangle contribution is independently clamped to
/// `[-(2π − δ), 2π − δ]`, where `δ = GWN_SOLID_ANGLE_CLIP`. A finite triangle's
/// solid angle has magnitude at most `2π`; therefore each clipped winding
/// contribution changes by at most `δ/(4π)`. With `k` clipped triangles, the
/// total change is at most `kδ/(4π)`. This controls only the change introduced
/// by the clamp. It does not bound arithmetic error, query distance, or
/// classification error, and it does not force a boundary query to return
/// one-half winding.
///
/// The ICCV 2025 bounded formulation cited below concerns point-cloud winding
/// numbers; it does not derive this triangle-mesh clamp.
///
/// ## Complexity — O(n) per query, identical to [`gwn_prepared`].
///
/// ## Reference
///
/// Reference: Koneputugodage et al., "Leaps and Bounds: An Improved Point Cloud
/// Winding Number Formulation for Fast Normal Estimation and Surface
/// Reconstruction," ICCV 2025 ([paper](https://openaccess.thecvf.com/content/ICCV2025/html/Koneputugodage_Leaps_and_Bounds_An_Improved_Point_Cloud_Winding_Number_Formulation_ICCV_2025_paper.html)).
/// The paper's formulation is for point clouds and is not an analysis of this
/// triangle-mesh clamp.
#[inline]
pub(crate) fn gwn_bounded_prepared(query: &Point3r, faces: &[PreparedFace]) -> f64 {
    let mut solid_angle_sum = 0.0_f64;
    let max_omega = 2.0 * std::f64::consts::PI - GWN_SOLID_ANGLE_CLIP;
    for face in faces {
        if let Some((va, vb, vc)) = vertex_offsets(query, face) {
            let omega = solid_angle_f64(va, vb, vc);
            solid_angle_sum += omega.clamp(-max_omega, max_omega);
        }
    }
    (solid_angle_sum / (4.0 * std::f64::consts::PI)).clamp(-1.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::storage::face_store::FaceData;
    use crate::infrastructure::storage::vertex_pool::VertexPool;

    fn unit_cube_mesh() -> (VertexPool, Vec<FaceData>) {
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

    #[test]
    fn gwn_unit_cube_interior_is_one() {
        let (pool, faces) = unit_cube_mesh();
        let wn = gwn::<f64>(&Point3r::new(0.0, 0.0, 0.0), &faces, &pool);
        assert!(
            (wn - 1.0).abs() < 0.02,
            "GWN at interior should be ≈1.0, got {wn:.4}"
        );
    }

    #[test]
    fn gwn_unit_cube_exterior_is_zero() {
        let (pool, faces) = unit_cube_mesh();
        let wn = gwn::<f64>(&Point3r::new(10.0, 0.0, 0.0), &faces, &pool);
        assert!(
            wn.abs() < 0.02,
            "GWN at exterior should be ≈0.0, got {wn:.4}"
        );
    }

    #[test]
    fn gwn_always_clamped_to_unit_interval() {
        let (pool, faces) = unit_cube_mesh();
        for (x, y, z) in [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (100.0, 100.0, 100.0),
        ] {
            let wn = gwn::<f64>(&Point3r::new(x, y, z), &faces, &pool);
            assert!(
                (-1.0..=1.0).contains(&wn),
                "GWN ({x},{y},{z}) out of [-1,1]: {wn}"
            );
        }
    }

    #[test]
    fn gwn_empty_mesh_is_zero() {
        let pool = VertexPool::default_millifluidic();
        assert_eq!(gwn::<f64>(&Point3r::new(0.0, 0.0, 0.0), &[], &pool), 0.0);
    }

    /// Regression: f32 guard must not produce NaN at vertex position.
    ///
    /// Old code used `T::from_f64(1e-40)` which flushed to `0.0_f32`,
    /// disabling the guard and allowing `atan2(0, 0) = NaN`.
    #[test]
    fn gwn_f32_near_vertex_does_not_nan() {
        let mut pool: VertexPool<f32> = VertexPool::<f32>::default_millifluidic();
        let n = leto::geometry::Vector3::<f32>::zeros();
        let v0 = pool.insert_or_weld(leto::geometry::Point3::new(0.0_f32, 0.0, 0.0), n);
        let v1 = pool.insert_or_weld(leto::geometry::Point3::new(1.0_f32, 0.0, 0.0), n);
        let v2 = pool.insert_or_weld(leto::geometry::Point3::new(0.0_f32, 1.0, 0.0), n);
        let faces = vec![FaceData::untagged(v0, v1, v2)];
        let query = leto::geometry::Point3::new(0.0_f32, 0.0, 0.0);
        let wn = gwn::<f32>(&query, &faces, &pool);
        assert!(
            wn.is_finite(),
            "GWN<f32> at vertex must be finite, got {wn}"
        );
        assert!(
            (-1.0..=1.0).contains(&wn),
            "GWN<f32> must be in [-1, 1], got {wn}"
        );
    }

    /// GWN on a zero-area degenerate triangle must be finite and in [-1,1].
    #[test]
    fn gwn_degenerate_zero_area_triangle_is_finite() {
        let mut pool = VertexPool::default_millifluidic();
        let n = leto::geometry::Vector3::zeros();
        let v0 = pool.insert_unique(Point3r::new(1.0, 0.0, 0.0), n);
        let v1 = pool.insert_unique(Point3r::new(1.0, 0.0, 0.0), n);
        let v2 = pool.insert_unique(Point3r::new(1.0, 0.0, 0.0), n);
        let faces = vec![FaceData::untagged(v0, v1, v2)];
        let wn = gwn::<f64>(&Point3r::new(0.0, 0.0, 0.0), &faces, &pool);
        assert!(
            wn.is_finite(),
            "GWN on degenerate face must be finite, got {wn}"
        );
        assert!(
            (-1.0..=1.0).contains(&wn),
            "GWN must be in [-1,1], got {wn}"
        );
    }

    /// GWN of far-exterior points is always ≈ 0 for a closed manifold.
    #[test]
    fn gwn_exterior_points_always_near_zero() {
        let (pool, faces) = unit_cube_mesh();
        for (x, y, z) in [
            (10.0, 0.0, 0.0),
            (-10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, -10.0, 0.0),
            (0.0, 0.0, 10.0),
            (5.0, 5.0, 5.0),
        ] {
            let wn = gwn::<f64>(&Point3r::new(x, y, z), &faces, &pool);
            assert!(
                wn.abs() < 0.1,
                "GWN at far exterior ({x},{y},{z}) should be ≈0, got {wn}"
            );
        }
    }
}
