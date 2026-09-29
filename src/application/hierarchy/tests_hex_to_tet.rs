//! Tests for the parent module, extracted from the module body.

    use super::canonical_tri_key;
    use super::estimated_face_map_capacity;
    use super::HexToTetConverter;
    use crate::domain::core::index::FaceId;
    use crate::domain::core::index::VertexId;
    use crate::domain::grid::StructuredHexGridBuilder;
    use crate::domain::mesh::IndexedMesh;
    use crate::domain::topology::{Cell, ElementType};

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

    #[test]
    fn adversarial_canonical_tri_key_is_orientation_invariant() {
        let a = VertexId::new(10);
        let b = VertexId::new(2);
        let c = VertexId::new(7);
        let k1 = canonical_tri_key([a, b, c]);
        let k2 = canonical_tri_key([c, b, a]);
        let k3 = canonical_tri_key([b, a, c]);
        assert_eq!(k1, k2);
        assert_eq!(k1, k3);
    }

    #[test]
    fn adversarial_neighbor_contains_matches_linear_membership() {
        let mut v = vec![
            VertexId::new(9),
            VertexId::new(1),
            VertexId::new(7),
            VertexId::new(3),
            VertexId::new(3),
            VertexId::new(2),
        ];
        v.sort_unstable_by_key(|id| id.as_usize());
        v.dedup();
        for probe in 0..12 {
            let p = VertexId::new(probe);
            let linear = v.contains(&p);
            let bounded = HexToTetConverter::contains_neighbor(&v, p);
            assert_eq!(
                bounded, linear,
                "bounded membership must match linear membership"
            );
        }
    }

    #[test]
    fn face_map_capacity_covers_mixed_cell_insertions() {
        let cells = vec![
            Cell::hexahedron(0, 1, 2, 3, 4, 5),
            Cell::tetrahedron(6, 7, 8, 9),
        ];

        assert_eq!(estimated_face_map_capacity(&cells), 6 * 4 + 4);
    }
