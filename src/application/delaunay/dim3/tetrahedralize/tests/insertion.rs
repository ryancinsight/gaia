//! Engine insertion: every non-degenerate input point survives.

#[cfg(test)]
mod tests {
    use crate::application::delaunay::dim3::tetrahedralize::BowyerWatson3D;
    use leto::geometry::Point3;

    fn sample_points() -> [Point3<f64>; 6] {
        [
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.9, 0.1, 0.1),
            Point3::new(0.1, 0.9, 0.1),
            Point3::new(0.1, 0.1, 0.9),
            Point3::new(0.72, 0.64, 0.58),
            Point3::new(0.38, 0.44, 0.31),
        ]
    }

    #[test]
    fn bowyer_watson_retains_all_non_degenerate_input_points() {
        let points = sample_points();
        let mut engine = BowyerWatson3D::with_capacity(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            points.len(),
        );
        for &point in &points {
            engine.insert_point(point);
        }

        let (vertices, tetrahedra) = engine.finalize();

        assert_eq!(vertices.len(), points.len());
        assert!(!tetrahedra.is_empty());
        for tet in tetrahedra {
            assert!(tet.iter().all(|&index| index < vertices.len()));
            assert!(tet
                .iter()
                .enumerate()
                .all(|(index, vertex)| !tet[..index].contains(vertex)));
            let [a, b, c, d] = tet.map(|index| vertices[index]);
            let six_volume = (b - a).cross(c - a).dot(d - a);
            assert!(six_volume.abs() > 1e-12, "degenerate tetrahedron: {tet:?}");
        }
    }
}
