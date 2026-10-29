//! Face clipping: keep, remove, split, coplanar, degenerate.

#[cfg(test)]
mod tests {
    use super::super::fixtures::face_from_pts;
    use super::super::fixtures::p;
    use crate::application::csg::clip::plane::clip_face_by_plane;
    use crate::application::csg::clip::plane::PlaneEquation;

    #[test]
    fn clip_fully_inside_keeps_face() {
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 2.0), p(0.0, 1.0, 3.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let result = clip_face_by_plane(&face, &mut pool, &plane);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], face);
    }

    #[test]
    fn clip_fully_outside_removes_face() {
        let (face, mut pool) =
            face_from_pts(p(0.0, 0.0, -1.0), p(1.0, 0.0, -2.0), p(0.0, 1.0, -3.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let result = clip_face_by_plane(&face, &mut pool, &plane);
        assert!(result.is_empty());
    }

    #[test]
    fn clip_straddling_produces_subtriangles() {
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 1.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let result = clip_face_by_plane(&face, &mut pool, &plane);
        assert!(
            !result.is_empty(),
            "straddling face should produce inside sub-faces"
        );
        // One vertex below → inside polygon is 4 vertices → 2 triangles
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn clip_coplanar_keeps_face() {
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let result = clip_face_by_plane(&face, &mut pool, &plane);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn clip_one_vertex_on_plane_two_above() {
        // v0 on plane, v1 and v2 above → fully inside (Degenerate counts as
        // inside).
        let (face, mut pool) = face_from_pts(p(0.5, 0.5, 0.0), p(1.0, 0.0, 1.0), p(0.0, 1.0, 1.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let result = clip_face_by_plane(&face, &mut pool, &plane);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn clip_face_degenerate_plane_no_panic() {
        // Degenerate plane (collinear points) → normal = 0.
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 1.0), p(0.0, 1.0, 1.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(2.0, 0.0, 0.0));
        // All orient_3d calls return Degenerate for a degenerate plane.
        let result = clip_face_by_plane(&face, &mut pool, &plane);
        // Should not panic; face treated as coplanar (all Degenerate).
        assert!(!result.is_empty());
    }
}
