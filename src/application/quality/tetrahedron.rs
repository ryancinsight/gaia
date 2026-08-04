//! Native-precision quality metrics for linear tetrahedral cells.
//!
//! The metrics are dimensionless except for volume.  They are evaluated in
//! the mesh scalar `T` and converted to `f64` only when a report is assembled,
//! so `IndexedMesh<f32>` does not silently execute its quality kernel in
//! double precision.

use leto::geometry::{Point3, Vector3};

use crate::application::quality::metrics::QualityMetric;
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::ElementType;

/// Per-tetrahedron quality values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetrahedronQuality<T> {
    /// Positive geometric volume.
    pub volume: T,
    /// Circumradius divided by the shortest edge length.
    pub radius_edge_ratio: T,
    /// Smallest interior dihedral angle in radians.
    pub min_dihedral_angle: T,
    /// Volume normalized so an equilateral tetrahedron has value one.
    pub normalized_volume: T,
}

/// Summary statistics for the tetrahedral cells in an indexed mesh.
#[derive(Clone, Debug)]
pub struct TetrahedralQualityReport {
    /// Volume statistics for valid tetrahedral cells.
    pub volume: Option<QualityMetric>,
    /// Radius-edge ratio statistics for valid tetrahedral cells.
    pub radius_edge_ratio: Option<QualityMetric>,
    /// Minimum-dihedral-angle statistics in radians for valid cells.
    pub min_dihedral_angle: Option<QualityMetric>,
    /// Normalized-volume statistics for valid cells.
    pub normalized_volume: Option<QualityMetric>,
    /// Number of tetrahedral cells with valid geometry.
    pub valid_cell_count: usize,
    /// Number of tetrahedral cells that could not be measured.
    pub invalid_cell_count: usize,
}

const EDGES: [(usize, usize, usize, usize); 6] = [
    (0, 1, 2, 3),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
    (1, 2, 0, 3),
    (1, 3, 0, 2),
    (2, 3, 0, 1),
];

#[inline]
fn finite_point<T: Scalar>(point: &Point3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(point.x)
        && <T as eunomia::NumericElement>::is_finite(point.y)
        && <T as eunomia::NumericElement>::is_finite(point.z)
}

#[inline]
fn finite_vector<T: Scalar>(vector: &Vector3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(vector.x)
        && <T as eunomia::NumericElement>::is_finite(vector.y)
        && <T as eunomia::NumericElement>::is_finite(vector.z)
}

/// Measure one tetrahedron in native scalar precision.
///
/// `None` means that a coordinate is non-finite, the tetrahedron is
/// degenerate, or a derived metric is non-finite.  The returned angle is the
/// interior dihedral angle at the worst edge; lower values indicate a sliver.
#[must_use]
pub fn tetrahedron_quality<T: Scalar>(points: [Point3<T>; 4]) -> Option<TetrahedronQuality<T>> {
    if points.iter().any(|point| !finite_point(point)) {
        return None;
    }

    let [a, b, c, d] = points;
    let u = b - a;
    let v = c - a;
    let w = d - a;
    let signed_six_volume = u.dot(v.cross(w));
    let zero = <T as eunomia::NumericElement>::ZERO;
    if !<T as eunomia::NumericElement>::is_finite(signed_six_volume) || signed_six_volume == zero {
        return None;
    }

    let six = <T as Scalar>::from_f64(6.0);
    let volume = signed_six_volume.abs() / six;
    let two = <T as eunomia::NumericElement>::ONE + <T as eunomia::NumericElement>::ONE;
    let center_numerator = v.cross(w) * u.norm_squared()
        + w.cross(u) * v.norm_squared()
        + u.cross(v) * w.norm_squared();
    let circumcenter = center_numerator / (signed_six_volume * two);
    let radius_squared = circumcenter.norm_squared();
    if !finite_vector(&center_numerator)
        || !finite_vector(&circumcenter)
        || !<T as eunomia::NumericElement>::is_finite(radius_squared)
        || radius_squared <= zero
    {
        return None;
    }

    let mut edge_squared = [zero; 6];
    let mut shortest_edge_squared = <T as eunomia::NumericElement>::INFINITY;
    for (index, &(i, j, _, _)) in EDGES.iter().enumerate() {
        let edge = points[j] - points[i];
        let squared = edge.norm_squared();
        if !<T as eunomia::NumericElement>::is_finite(squared) || squared <= zero {
            return None;
        }
        edge_squared[index] = squared;
        shortest_edge_squared = shortest_edge_squared.min_scalar(squared);
    }

    let radius_edge_ratio = radius_squared.sqrt() / shortest_edge_squared.sqrt();
    let mut min_dihedral_angle = <T as eunomia::NumericElement>::INFINITY;
    for &(i, j, k, l) in &EDGES {
        let edge = points[j] - points[i];
        let edge_squared = edge.norm_squared();
        let face_k = points[k] - points[i];
        let face_l = points[l] - points[i];
        let projected_k = face_k - edge * (face_k.dot(edge) / edge_squared);
        let projected_l = face_l - edge * (face_l.dot(edge) / edge_squared);
        let product = projected_k.norm_squared() * projected_l.norm_squared();
        if product <= zero || !<T as eunomia::NumericElement>::is_finite(product) {
            return None;
        }
        let cosine = (projected_k.dot(projected_l) / product.sqrt())
            .max_scalar(-<T as eunomia::NumericElement>::ONE)
            .min_scalar(<T as eunomia::NumericElement>::ONE);
        min_dihedral_angle = min_dihedral_angle.min_scalar(cosine.acos());
    }

    let edge_sum = edge_squared
        .iter()
        .copied()
        .fold(zero, |sum, squared| sum + squared);
    let edge_mean_squared = edge_sum / six;
    let edge_rms = edge_mean_squared.sqrt();
    let normalized_volume = ((six + six) * volume)
        / (<T as Scalar>::from_f64(2.0).sqrt() * edge_rms * edge_mean_squared);
    if !<T as eunomia::NumericElement>::is_finite(radius_edge_ratio)
        || !<T as eunomia::NumericElement>::is_finite(min_dihedral_angle)
        || !<T as eunomia::NumericElement>::is_finite(normalized_volume)
    {
        return None;
    }

    Some(TetrahedronQuality {
        volume,
        radius_edge_ratio,
        min_dihedral_angle,
        normalized_volume,
    })
}

/// Summarize native-precision quality metrics for tetrahedral cells.
///
/// Non-tetrahedral cells are ignored.  A malformed or degenerate tetrahedron
/// increments `invalid_cell_count`; it is not converted into a default metric.
/// `None` is returned when the mesh contains no tetrahedral cells.
#[must_use]
pub fn tetrahedral_quality_report<T: Scalar>(
    mesh: &IndexedMesh<T>,
) -> Option<TetrahedralQualityReport> {
    let tetrahedral_cell_count = mesh
        .cells
        .iter()
        .filter(|cell| cell.element_type == ElementType::Tetrahedron)
        .count();
    if tetrahedral_cell_count == 0 {
        return None;
    }

    let mut volumes = Vec::with_capacity(tetrahedral_cell_count);
    let mut radius_edge_ratios = Vec::with_capacity(tetrahedral_cell_count);
    let mut min_dihedral_angles = Vec::with_capacity(tetrahedral_cell_count);
    let mut normalized_volumes = Vec::with_capacity(tetrahedral_cell_count);
    let mut invalid_cell_count = 0;

    for cell in &mesh.cells {
        if cell.element_type != ElementType::Tetrahedron {
            continue;
        }
        let vertex_ids: [usize; 4] = if let Ok(vertex_ids) = cell.vertex_ids.as_slice().try_into() {
            vertex_ids
        } else {
            invalid_cell_count += 1;
            continue;
        };
        if vertex_ids.iter().any(|&id| id >= mesh.vertices.len()) {
            invalid_cell_count += 1;
            continue;
        }
        let points = vertex_ids.map(|id| {
            *mesh
                .vertices
                .position(crate::domain::core::index::VertexId::from_usize(id))
        });
        let Some(quality) = tetrahedron_quality(points) else {
            invalid_cell_count += 1;
            continue;
        };
        volumes.push(quality.volume);
        radius_edge_ratios.push(quality.radius_edge_ratio);
        min_dihedral_angles.push(quality.min_dihedral_angle);
        normalized_volumes.push(quality.normalized_volume);
    }

    Some(TetrahedralQualityReport {
        volume: QualityMetric::from_scalar_values(&volumes),
        radius_edge_ratio: QualityMetric::from_scalar_values(&radius_edge_ratios),
        min_dihedral_angle: QualityMetric::from_scalar_values(&min_dihedral_angles),
        normalized_volume: QualityMetric::from_scalar_values(&normalized_volumes),
        valid_cell_count: volumes.len(),
        invalid_cell_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::TetrahedralMeshBuilder;

    fn analytical_quality<T: Scalar>() -> TetrahedronQuality<T> {
        let scalar = |value| <T as Scalar>::from_f64(value);
        tetrahedron_quality([
            Point3::new(scalar(0.0), scalar(0.0), scalar(0.0)),
            Point3::new(scalar(1.0), scalar(0.0), scalar(0.0)),
            Point3::new(scalar(0.0), scalar(1.0), scalar(0.0)),
            Point3::new(scalar(0.0), scalar(0.0), scalar(1.0)),
        ])
        .expect("analytical tetrahedron is non-degenerate")
    }

    fn assert_analytical_quality<T: Scalar>() {
        let quality = analytical_quality::<T>();
        assert!((quality.volume.to_f64() - 1.0 / 6.0).abs() < 1e-6);
        assert!((quality.radius_edge_ratio.to_f64() - 3.0_f64.sqrt() / 2.0).abs() < 1e-6);
        assert!((quality.normalized_volume.to_f64() - 0.769800358919501).abs() < 1e-6);
        assert!(quality.min_dihedral_angle.to_f64().to_degrees() > 54.0);
    }

    #[test]
    fn analytical_tetrahedron_has_expected_native_metrics() {
        assert_analytical_quality::<f32>();
        assert_analytical_quality::<f64>();
    }

    #[test]
    fn sliver_has_worse_dihedral_and_normalized_volume() {
        let regular = analytical_quality::<f64>();
        let sliver = tetrahedron_quality([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.45, 0.45, 1e-4),
        ])
        .expect("sliver remains non-degenerate");
        assert!(sliver.min_dihedral_angle < regular.min_dihedral_angle);
        assert!(sliver.normalized_volume < regular.normalized_volume);
        assert!(sliver.radius_edge_ratio > regular.radius_edge_ratio);
    }

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
