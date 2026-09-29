//! Accelerated plane-based clip and refine operations.
//!
//! # CGAL 6.1 Insight — Plane Clippers (2025)
//!
//! CGAL 6.1 reimplemented `clip()`, `split()`, and introduced
//! `refine_with_plane()` achieving 10× speedup by:
//!
//! 1. **Cached plane equations** — compute normal + offset once, reuse for all
//!    orientation tests against that plane.
//! 2. **Batch vertex classification** — classify every unique vertex against the
//!    cut plane in a single O(V) pass before processing faces, reducing
//!    redundant `orient_3d` calls from O(3F) to O(V).
//! 3. **Direct plane clipping** — split straddling faces at the plane directly,
//!    producing 1–3 sub-triangles without the overhead of full CDT
//!    co-refinement.
//!
//! These techniques prevent cascading degenerates (thin walls from repeated
//! subdivision) and complement corefinement for hybrid workflows (e.g.,
//! pre-split at symmetry planes before full Boolean).
//!
//! # Theorem — Batch Vertex Classification Equivalence
//!
//! Let V be the set of unique vertex IDs referenced by F faces, and let P be a
//! plane defined by three CCW points.  Per-face classification calls
//! `orient_3d` 3|F| times, but since vertices are shared, |V| ≤ 3|F| with
//! equality only for isolated triangles.  Batch classification calls
//! `orient_3d` exactly |V| times.  Since `orient_3d` is a pure function of its
//! arguments, the per-vertex results are identical in both approaches.  ∎
//!
//! # References
//!
//! - Sébastien Loriot, Mael Rouxel-Labbé, Jane Tournois, Ilker O. Yaz,
//!   "Polygon Mesh Processing", CGAL 6.1, 2025.

use hashbrown::HashMap;

use crate::domain::core::index::VertexId;
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
    pub fn from_points(pa: &Point3r, pb: &Point3r, pc: &Point3r) -> Self {
        let pa_arr = [pa.x, pa.y, pa.z];
        let pb_arr = [pb.x, pb.y, pb.z];
        let pc_arr = [pc.x, pc.y, pc.z];

        let ab = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
        let ac = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let offset = n[0] * pa.x + n[1] * pa.y + n[2] * pa.z;

        Self {
            pa: pa_arr,
            pb: pb_arr,
            pc: pc_arr,
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

/// Classify a face against a cached plane equation.
///
/// Returns the classification and the three per-vertex orientations.
#[must_use]
pub fn classify_face(
    face: &FaceData,
    pool: &VertexPool,
    plane: &PlaneEquation,
) -> (FacePlaneClass, [Orientation; 3]) {
    let signs = [
        plane.classify(pool.position(face.vertices[0])),
        plane.classify(pool.position(face.vertices[1])),
        plane.classify(pool.position(face.vertices[2])),
    ];

    let any_pos = signs.contains(&Orientation::Positive);
    let any_neg = signs.contains(&Orientation::Negative);

    let class = match (any_pos, any_neg) {
        (true, true) => FacePlaneClass::Straddling,
        (true, false) => FacePlaneClass::Inside,
        (false, true) => FacePlaneClass::Outside,
        (false, false) => FacePlaneClass::Coplanar,
    };

    (class, signs)
}

// ── Face-level plane clipping ─────────────────────────────────────────────────

/// Clip a face by a plane, returning sub-faces in the positive half-space.
///
/// Uses exact `orient_3d` for inside/outside classification and float
/// arithmetic for intersection positions (acceptable per Shewchuk: position
/// errors do not affect topology).
///
/// New vertices from edge–plane intersections are inserted into `pool` via
/// `insert_or_weld`.
///
/// Returns 0 faces if fully clipped, 1 face if fully kept, or 1–2 faces for
/// a straddling triangle.
pub fn clip_face_by_plane(
    face: &FaceData,
    pool: &mut VertexPool,
    plane: &PlaneEquation,
) -> Vec<FaceData> {
    let (class, signs) = classify_face(face, pool, plane);

    match class {
        FacePlaneClass::Inside | FacePlaneClass::Coplanar => vec![*face],
        FacePlaneClass::Outside => Vec::new(),
        FacePlaneClass::Straddling => {
            let vids = face.vertices;
            let positions = [
                *pool.position(vids[0]),
                *pool.position(vids[1]),
                *pool.position(vids[2]),
            ];
            let plane_n = plane.normal_vec();

            let mut output: Vec<VertexId> = Vec::with_capacity(4);
            for i in 0..3 {
                let j = (i + 1) % 3;
                let s_in = signs[i] != Orientation::Negative;
                let e_in = signs[j] != Orientation::Negative;

                match (s_in, e_in) {
                    (true, true) => output.push(vids[j]),
                    (true, false) => {
                        let cut = plane.intersect_edge(&positions[i], &positions[j]);
                        output.push(pool.insert_or_weld(cut, plane_n));
                    }
                    (false, true) => {
                        let cut = plane.intersect_edge(&positions[i], &positions[j]);
                        output.push(pool.insert_or_weld(cut, plane_n));
                        output.push(vids[j]);
                    }
                    (false, false) => {}
                }
            }
            fan_triangulate_vids(&output, face.region)
        }
    }
}

// ── Mesh-level plane refinement ───────────────────────────────────────────────

/// Split all faces of a mesh by a plane.
///
/// # CGAL 6.1 — `refine_with_plane`
///
/// Equivalent to CGAL 6.1's `refine_with_plane()`:
///
/// 1. **Batch classify** all unique vertices against the plane — O(V)
///    `orient_3d` calls (each vertex tested exactly once).
/// 2. For each face, look up cached per-vertex classifications — O(1).
/// 3. Non-straddling faces pass through unchanged.
/// 4. Straddling faces are split at the plane, producing 1–3 sub-faces per
///    side.
///
/// Total: O(V + F_straddling) instead of O(3F) for naive per-face orient_3d.
///
/// Returns `(inside_faces, outside_faces)` — both are valid face soups using
/// the same pool.  Coplanar faces are included in **both** halves.
pub fn refine_faces_with_plane(
    faces: &[FaceData],
    pool: &mut VertexPool,
    plane: &PlaneEquation,
) -> (Vec<FaceData>, Vec<FaceData>) {
    // Phase 1: batch-classify all unique vertices.
    let mut vertex_signs: HashMap<VertexId, Orientation> = HashMap::with_capacity(faces.len() * 2);
    for face in faces {
        for &vid in &face.vertices {
            vertex_signs
                .entry(vid)
                .or_insert_with(|| plane.classify(pool.position(vid)));
        }
    }

    let mut inside = Vec::with_capacity(faces.len());
    let mut outside = Vec::with_capacity(faces.len() / 4);

    // Phase 2: process each face using cached classifications.
    for face in faces {
        let signs = [
            vertex_signs[&face.vertices[0]],
            vertex_signs[&face.vertices[1]],
            vertex_signs[&face.vertices[2]],
        ];
        let any_pos = signs.contains(&Orientation::Positive);
        let any_neg = signs.contains(&Orientation::Negative);

        match (any_pos, any_neg) {
            (true, false) => inside.push(*face),
            (false, true) => outside.push(*face),
            (false, false) => {
                inside.push(*face);
                outside.push(*face);
            }
            (true, true) => {
                let (pos, neg) = split_straddling_face(face, pool, plane, &signs);
                inside.extend(pos);
                outside.extend(neg);
            }
        }
    }

    (inside, outside)
}

/// Split a straddling face into positive-side and negative-side sub-faces.
///
/// Walks the triangle edges in order, tracking which vertices/intersection
/// points belong to each half.  Intersection points are added to both polygons.
fn split_straddling_face(
    face: &FaceData,
    pool: &mut VertexPool,
    plane: &PlaneEquation,
    signs: &[Orientation; 3],
) -> (Vec<FaceData>, Vec<FaceData>) {
    let vids = face.vertices;
    let positions = [
        *pool.position(vids[0]),
        *pool.position(vids[1]),
        *pool.position(vids[2]),
    ];
    let plane_n = plane.normal_vec();

    let mut pos_poly: Vec<VertexId> = Vec::with_capacity(4);
    let mut neg_poly: Vec<VertexId> = Vec::with_capacity(4);

    for i in 0..3 {
        let j = (i + 1) % 3;

        // Emit vertex to its polygon.
        match signs[i] {
            Orientation::Positive => pos_poly.push(vids[i]),
            Orientation::Negative => neg_poly.push(vids[i]),
            Orientation::Degenerate => {
                pos_poly.push(vids[i]);
                neg_poly.push(vids[i]);
            }
        }

        // If the edge crosses the plane, emit intersection to both polygons.
        let crosses = (signs[i] == Orientation::Positive && signs[j] == Orientation::Negative)
            || (signs[i] == Orientation::Negative && signs[j] == Orientation::Positive);
        if crosses {
            let cut = plane.intersect_edge(&positions[i], &positions[j]);
            let cut_vid = pool.insert_or_weld(cut, plane_n);
            pos_poly.push(cut_vid);
            neg_poly.push(cut_vid);
        }
    }

    let region = face.region;
    (
        fan_triangulate_vids(&pos_poly, region),
        fan_triangulate_vids(&neg_poly, region),
    )
}

/// Fan-triangulate a convex polygon given as vertex IDs.
fn fan_triangulate_vids(
    vids: &[VertexId],
    region: crate::domain::core::index::RegionId,
) -> Vec<FaceData> {
    if vids.len() < 3 {
        return Vec::new();
    }
    let root = vids[0];
    (1..vids.len() - 1)
        .map(|k| FaceData::new(root, vids[k], vids[k + 1], region))
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_plane.rs"]
mod tests;
