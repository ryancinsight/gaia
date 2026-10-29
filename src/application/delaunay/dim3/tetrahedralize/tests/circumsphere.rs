//! Circumsphere predicate: strictly external points are rejected.

#[cfg(test)]
mod tests {
    use crate::application::delaunay::dim3::tetrahedralize::Tetrahedron;
    use leto::geometry::Point3;

    #[test]
    fn tetrahedron_circumsphere_rejects_a_strictly_external_point() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let tetrahedron = Tetrahedron::new([0, 1, 2, 3], &points);
        let external = Point3::new(3.0, 3.0, 3.0);

        assert!(!tetrahedron.contains_in_circumsphere(&external, &points));
    }

    // ── GAIA-002: native-precision f32 oracle ────────────────────────────────
}
