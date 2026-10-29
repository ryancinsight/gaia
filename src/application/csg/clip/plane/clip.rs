//! Face-level plane clipping: classification, splitting, fan repair.

use crate::domain::core::index::VertexId;
use crate::domain::geometry::predicates::Orientation;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

use super::equation::{FacePlaneClass, PlaneEquation};

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

/// Split a straddling face into positive-side and negative-side sub-faces.
///
/// Walks the triangle edges in order, tracking which vertices/intersection
/// points belong to each half.  Intersection points are added to both polygons.
pub(super) fn split_straddling_face(
    face: &FaceData,
    pool: &mut VertexPool,
    plane: &PlaneEquation,
    signs: [Orientation; 3],
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
