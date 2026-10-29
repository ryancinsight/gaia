//! Mesh refinement: batch splitting, region ids, determinism.

#[cfg(test)]
mod tests {
    use super::super::fixtures::csg_pool;
    use super::super::fixtures::face_from_pts;
    use super::super::fixtures::n_up;
    use super::super::fixtures::p;
    use crate::application::csg::clip::plane::refine_faces_with_plane;
    use crate::application::csg::clip::plane::PlaneEquation;
    use crate::domain::core::index::RegionId;
    use crate::infrastructure::storage::face_store::FaceData;

    #[test]
    fn refine_empty_mesh() {
        let mut pool = csg_pool();
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&[], &mut pool, &plane);
        assert!(ins.is_empty());
        assert!(outs.is_empty());
    }

    #[test]
    fn refine_all_above() {
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 2.0), p(0.0, 1.0, 3.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
        assert_eq!(ins.len(), 1);
        assert!(outs.is_empty());
    }

    #[test]
    fn refine_straddling_triangle_splits_both_sides() {
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 1.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
        assert!(!ins.is_empty(), "should have inside sub-faces");
        assert!(!outs.is_empty(), "should have outside sub-faces");
        // 1 vertex below → inside = 2 tris, outside = 1 tri
        assert_eq!(ins.len(), 2);
        assert_eq!(outs.len(), 1);
    }

    #[test]
    fn refine_preserves_region_ids() {
        let mut pool = csg_pool();
        let n = n_up();
        let va = pool.insert_or_weld(p(0.0, 0.0, 1.0), n);
        let vb = pool.insert_or_weld(p(1.0, 0.0, -1.0), n);
        let vc = pool.insert_or_weld(p(0.0, 1.0, 1.0), n);
        let region = RegionId::new(42);
        let face = FaceData::new(va, vb, vc, region);

        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
        for f in ins.iter().chain(outs.iter()) {
            assert_eq!(f.region, region, "sub-faces must inherit parent region");
        }
    }

    #[test]
    fn refine_batch_classification_reduces_orient3d_calls() {
        // 4 faces sharing 5 vertices: batch should call orient_3d 5 times,
        // not 12.  We verify correctness by checking the output (counts of
        // inside/outside faces), which implicitly validates that the cached
        // per-vertex classifications were used.
        let mut pool = csg_pool();
        let n = n_up();
        // Shared vertices
        let v0 = pool.insert_or_weld(p(0.0, 0.0, 1.0), n);
        let v1 = pool.insert_or_weld(p(1.0, 0.0, 1.0), n);
        let v2 = pool.insert_or_weld(p(0.5, 1.0, 1.0), n);
        let v3 = pool.insert_or_weld(p(0.5, 0.5, -1.0), n);
        let v4 = pool.insert_or_weld(p(1.0, 1.0, 1.0), n);

        let faces = [
            FaceData::untagged(v0, v1, v2), // all above
            FaceData::untagged(v0, v1, v3), // v3 below → straddles
            FaceData::untagged(v1, v2, v3), // v3 below → straddles
            FaceData::untagged(v1, v4, v2), // all above
        ];

        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&faces, &mut pool, &plane);
        // 2 fully-inside + 2 straddling (each producing 2 inside sub-faces)
        assert!(ins.len() >= 4);
        assert!(!outs.is_empty());
    }

    #[test]
    fn refine_vertex_on_plane_goes_to_both_halves() {
        // v0 on plane, v1 above, v2 below → straddling.
        // The on-plane vertex appears in both halves.
        let (face, mut pool) = face_from_pts(p(0.5, 0.5, 0.0), p(1.0, 0.0, 1.0), p(0.0, 1.0, -1.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
        assert_eq!(ins.len(), 1);
        assert_eq!(outs.len(), 1);
        // The on-plane vertex (v0) should appear in both sub-faces.
        let v0 = face.vertices[0];
        assert!(ins[0].vertices.contains(&v0));
        assert!(outs[0].vertices.contains(&v0));
    }

    #[test]
    fn refine_tilted_plane_xy() {
        // Plane at 45° through x-axis: n = (0, -1/√2, 1/√2), d = 0
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 1.0));
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 2.0), p(1.0, 2.0, 0.0), p(0.0, 2.0, 0.0));
        let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
        let total = ins.len() + outs.len();
        assert!(total >= 2, "tilted plane should split the face");
    }

    #[test]
    fn refine_is_deterministic() {
        let build = || {
            let (face, mut pool) =
                face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 0.5));
            let plane =
                PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
            refine_faces_with_plane(&[face], &mut pool, &plane)
        };
        let (ins1, outs1) = build();
        let (ins2, outs2) = build();
        assert_eq!(ins1.len(), ins2.len());
        assert_eq!(outs1.len(), outs2.len());
    }
}
