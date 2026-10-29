//! Hex-to-tet properties: key invariance, membership, capacity.

#[cfg(test)]
mod tests {
    use crate::application::hierarchy::hex_to_tet::convert::{
        canonical_tri_key, estimated_face_map_capacity,
    };
    use crate::application::hierarchy::hex_to_tet::HexToTetConverter;
    use crate::domain::core::index::VertexId;
    use crate::domain::topology::Cell;

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
}
