//! Face classification: above, below, straddling, coplanar.

#[cfg(test)]
mod tests {
    use super::super::fixtures::face_from_pts;
    use super::super::fixtures::p;
    use crate::application::csg::clip::plane::classify_face;
    use crate::application::csg::clip::plane::FacePlaneClass;
    use crate::application::csg::clip::plane::PlaneEquation;

    #[test]
    fn classify_face_above_plane() {
        let (face, pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 2.0), p(0.0, 1.0, 3.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (class, _) = classify_face(&face, &pool, &plane);
        assert_eq!(class, FacePlaneClass::Inside);
    }

    #[test]
    fn classify_face_below_plane() {
        let (face, pool) = face_from_pts(p(0.0, 0.0, -1.0), p(1.0, 0.0, -2.0), p(0.0, 1.0, -3.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (class, _) = classify_face(&face, &pool, &plane);
        assert_eq!(class, FacePlaneClass::Outside);
    }

    #[test]
    fn classify_face_straddling() {
        let (face, pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 0.5));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (class, _) = classify_face(&face, &pool, &plane);
        assert_eq!(class, FacePlaneClass::Straddling);
    }

    #[test]
    fn classify_face_coplanar() {
        let (face, pool) = face_from_pts(p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        let (class, _) = classify_face(&face, &pool, &plane);
        assert_eq!(class, FacePlaneClass::Coplanar);
    }
}
