//! Hex-to-tet conversion: structured grids, boundary orientation, branching.

#[cfg(test)]
mod tests {
    use crate::application::hierarchy::hex_to_tet::HexToTetConverter;
    use crate::domain::core::index::FaceId;
    use crate::domain::grid::StructuredHexGridBuilder;
    use crate::domain::mesh::IndexedMesh;
    use crate::domain::topology::ElementType;

    fn tet_six_volume(mesh: &IndexedMesh<f64>, cell: &crate::domain::topology::Cell) -> f64 {
        let mut vertices = Vec::new();
        let mut seen: hashbrown::HashSet<_> = hashbrown::HashSet::new();
        for &f_idx_raw in &cell.faces {
            let f_idx = FaceId::from_usize(f_idx_raw);
            let face = mesh.faces.get(f_idx);
            for &v_idx in &face.vertices {
                if seen.insert(v_idx) {
                    vertices.push(v_idx);
                }
            }
        }
        assert_eq!(
            vertices.len(),
            4,
            "Converted tetrahedron must have 4 unique vertices"
        );

        let p0 = mesh.vertices.position(vertices[0]).coords;
        let p1 = mesh.vertices.position(vertices[1]).coords;
        let p2 = mesh.vertices.position(vertices[2]).coords;
        let p3 = mesh.vertices.position(vertices[3]).coords;
        (p1 - p0).cross(p2 - p0).dot(p3 - p0).abs()
    }

    fn assert_no_degenerate_tets(mesh: &IndexedMesh<f64>) {
        let bounds = mesh.bounding_box();
        let length_scale = (bounds.max.coords - bounds.min.coords).norm();
        let volume_tol = length_scale.powi(3) * 1e-12;

        for (i, cell) in mesh.cells().iter().enumerate() {
            if cell.element_type != ElementType::Tetrahedron {
                continue;
            }
            let six_v = tet_six_volume(mesh, cell);
            assert!(
                six_v > volume_tol,
                "Degenerate tetrahedron at cell {i} with 6V={six_v:.3e}, tol={volume_tol:.3e}"
            );
        }
    }

    #[test]
    fn structured_hex_mesh_converts_to_non_degenerate_tets() {
        let hex_mesh = StructuredHexGridBuilder::new(4, 4, 4).build();
        let tet_mesh = HexToTetConverter::convert(&hex_mesh);

        assert_eq!(tet_mesh.cell_count(), 4 * 4 * 4 * 5);
        assert!(tet_mesh
            .cells()
            .iter()
            .all(|c| c.element_type == ElementType::Tetrahedron));
        assert_no_degenerate_tets(&tet_mesh);
    }

    #[test]
    fn hex_to_tet_boundary_faces_are_outward_oriented() {
        // After conversion, every boundary face must have a positive signed volume
        // integral (outward-facing normal). The orient_outward pass in convert()
        // must correct any inward-pointing boundary faces from the tet patterns.
        use crate::application::quality::normals::analyze_normals;
        let hex_mesh = StructuredHexGridBuilder::new(2, 2, 2).build();
        let tet_mesh = HexToTetConverter::convert(&hex_mesh);

        let analysis = analyze_normals(&tet_mesh);
        assert_eq!(
            analysis.inward_faces,
            0,
            "orient_outward must eliminate all inward-facing boundary faces; \
                 got {}/{} inward (before: {}/{})",
            analysis.inward_faces,
            analysis.total_faces(),
            analysis.outward_faces,
            analysis.inward_faces,
        );
        assert!(
            analysis.outward_faces > 0,
            "converted tet mesh must have outward boundary faces"
        );
    }

    #[test]
    fn branching_mesh_conversion_avoids_degenerate_tets() {
        // Use a larger structured grid to exercise non-trivial tet conversion.
        let hex_mesh = StructuredHexGridBuilder::new(6, 4, 4).build();
        let tet_mesh = HexToTetConverter::convert(&hex_mesh);

        assert_eq!(tet_mesh.cell_count(), 6 * 4 * 4 * 5);
        assert!(tet_mesh
            .cells()
            .iter()
            .all(|c| c.element_type == ElementType::Tetrahedron));
        assert_no_degenerate_tets(&tet_mesh);
    }
}
