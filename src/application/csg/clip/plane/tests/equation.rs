//! Plane equation: classification, construction, edge intersection.

#[cfg(test)]
mod tests {
    use super::super::fixtures::csg_pool;
    use super::super::fixtures::n_up;
    use super::super::fixtures::p;
    use crate::application::csg::clip::plane::PlaneEquation;
    use crate::domain::core::index::RegionId;
    use crate::domain::geometry::predicates::Orientation;
    use crate::infrastructure::storage::face_store::FaceData;

    #[test]
    fn plane_classify_above_below_on() {
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        assert_eq!(plane.classify(&p(0.5, 0.5, 1.0)), Orientation::Positive);
        assert_eq!(plane.classify(&p(0.5, 0.5, -1.0)), Orientation::Negative);
        assert_eq!(plane.classify(&p(0.5, 0.5, 0.0)), Orientation::Degenerate);
    }

    #[test]
    fn plane_from_face_matches_from_points() {
        let mut pool = csg_pool();
        let n = n_up();
        let va = pool.insert_or_weld(p(0.0, 0.0, 0.0), n);
        let vb = pool.insert_or_weld(p(1.0, 0.0, 0.0), n);
        let vc = pool.insert_or_weld(p(0.0, 1.0, 0.0), n);
        let face = FaceData::new(va, vb, vc, RegionId::INVALID);

        let p1 = PlaneEquation::from_face(&face, &pool);
        let p2 =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));

        let q = p(0.3, 0.2, 0.7);
        assert_eq!(p1.classify(&q), p2.classify(&q));
    }

    #[test]
    fn plane_intersect_edge_midpoint() {
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let s = p(0.0, 0.0, 1.0);
        let e = p(0.0, 0.0, -1.0);
        let cut = plane.intersect_edge(&s, &e);
        assert!((cut.x).abs() < 1e-12);
        assert!((cut.y).abs() < 1e-12);
        assert!((cut.z).abs() < 1e-12);
    }

    #[test]
    fn plane_intersect_edge_quarter() {
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 1.0), &p(1.0, 0.0, 1.0), &p(0.0, 1.0, 1.0));
        let s = p(0.0, 0.0, 0.0);
        let e = p(0.0, 0.0, 4.0);
        let cut = plane.intersect_edge(&s, &e);
        assert!((cut.z - 1.0).abs() < 1e-12);
    }
}
