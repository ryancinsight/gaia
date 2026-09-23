//! Behavioral tests for winding-number normal consistency.

#[cfg(test)]
mod tests {
    use crate::application::csg::arrangement::classify::{
        prepare_classification_faces, wnnc_score,
    };
    use crate::application::csg::arrangement::gwn_robustness_tests::tests::{
        sphere_mesh, unit_cube_mesh,
    };

    /// For a closed manifold, outward normals align with the winding gradient.
    #[test]
    fn wnnc_closed_cube_normals_consistent() {
        let (pool, faces) = unit_cube_mesh();
        let prepared = prepare_classification_faces(&faces, &pool);
        for face in &prepared {
            let face_length = face.normal.norm();
            if face_length < 1e-15 {
                continue;
            }
            let unit_normal = face.normal / face_length;
            let score = wnnc_score(&face.centroid, &unit_normal, &prepared);
            assert!(
                score > 0.0,
                "WNNC score should be positive for outward normal, got {score:.4} \
                 at centroid {centroid:?}",
                centroid = face.centroid
            );
        }
    }

    /// Reversing an outward normal reverses the WNNC score.
    #[test]
    fn wnnc_flipped_normals_negative() {
        let (pool, faces) = unit_cube_mesh();
        let prepared = prepare_classification_faces(&faces, &pool);
        let face = &prepared[0];
        let unit_normal = face.normal / face.normal.norm();
        let score = wnnc_score(&face.centroid, &-unit_normal, &prepared);
        assert!(
            score < 0.0,
            "WNNC score should be negative for flipped normal, got {score:.4}"
        );
    }

    /// Tessellated sphere face normals remain consistent with the winding field.
    #[test]
    fn wnnc_sphere_normals_consistent() {
        let (pool, faces) = sphere_mesh(8);
        let prepared = prepare_classification_faces(&faces, &pool);
        let mut positive_count = 0;
        let mut total_checked = 0;
        for face in &prepared {
            let normal_length = face.normal.norm();
            if normal_length < 1e-15 {
                continue;
            }
            let unit_normal = face.normal / normal_length;
            if wnnc_score(&face.centroid, &unit_normal, &prepared) > 0.0 {
                positive_count += 1;
            }
            total_checked += 1;
        }
        let ratio = f64::from(positive_count) / f64::from(total_checked);
        assert!(
            ratio > 0.9,
            "WNNC: only {positive_count}/{total_checked} ({percent:.1}%) faces consistent",
            percent = ratio * 100.0
        );
    }
}
