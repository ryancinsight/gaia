//! Mesh-level plane refinement: batch classify, split, and separate halves.

// ── Mesh-level plane refinement ───────────────────────────────────────────────

use hashbrown::HashMap;

use crate::domain::core::index::VertexId;
use crate::domain::geometry::predicates::Orientation;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

use super::clip::split_straddling_face;
use super::equation::PlaneEquation;

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
/// Total: O(V + `F_straddling`) instead of O(3F) for naive per-face `orient_3d`.
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
                let (pos, neg) = split_straddling_face(face, pool, plane, signs);
                inside.extend(pos);
                outside.extend(neg);
            }
        }
    }

    (inside, outside)
}
