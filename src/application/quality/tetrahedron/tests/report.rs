//! Report assembly: invalid cells counted, never defaulted.

#[cfg(test)]
mod tests {
    use crate::application::quality::tetrahedron::tetrahedral_quality_report;
    use crate::domain::mesh::TetrahedralMeshBuilder;

    #[test]
    fn report_counts_invalid_tetrahedra_without_defaulting_metrics() {
        let mut builder = TetrahedralMeshBuilder::<f64>::new();
        let a = builder.vertex_xyz(0.0, 0.0, 0.0);
        let b = builder.vertex_xyz(1.0, 0.0, 0.0);
        let c = builder.vertex_xyz(0.0, 1.0, 0.0);
        let d = builder.vertex_xyz(0.0, 0.0, 1.0);
        builder
            .tetrahedron([a, b, c, d])
            .expect("valid tetrahedron");
        let mut mesh = builder.build();
        mesh.cells
            .push(crate::domain::topology::Cell::tetrahedron(0, 0, 0, 0));
        let report = tetrahedral_quality_report(&mesh).expect("tetrahedral cells exist");
        assert_eq!(report.valid_cell_count, 1);
        assert_eq!(report.invalid_cell_count, 1);
        assert_eq!(report.volume.expect("valid metric").count, 1);
    }
}
