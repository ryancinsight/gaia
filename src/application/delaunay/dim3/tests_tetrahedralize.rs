//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::index::VertexId;
use crate::domain::geometry::predicates::{insphere, orient_3d, Orientation};
use crate::domain::mesh::TetrahedralMeshBuilder;

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

/// Dyadic fixture: every coordinate is a multiple of `2^-4`, so the same
/// values store identically at `f32` and `f64`. No five points are
/// co-spherical — the strict-Delaunay assertions below verify that
/// premise on the fixture.
fn dyadic_points() -> [Point3<f64>; 8] {
    [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.5, 0.4375, 0.5625),
        Point3::new(0.8125, 0.1875, 0.3125),
        Point3::new(0.1875, 0.75, 0.375),
        Point3::new(0.3125, 0.25, 0.8125),
    ]
}

/// Run the kernel over the unit box at precision `T` and strip the
/// super-tetrahedron anchors.
fn run_bowyer_watson<T: Scalar>(points: &[Point3<T>]) -> (Vec<Point3<T>>, Vec<[usize; 4]>) {
    let origin = <T as Scalar>::from_f64;
    let min = Point3::new(origin(0.0), origin(0.0), origin(0.0));
    let max = Point3::new(origin(1.0), origin(1.0), origin(1.0));
    let mut engine = BowyerWatson3D::with_capacity(min, max, points.len());
    for &point in points {
        engine.insert_point(point);
    }
    engine.finalize()
}

/// GAIA-002 acceptance oracle (a): an `IndexedMesh<f32>` tetrahedralization
/// satisfies the empty-circumsphere property at its own stored precision.
///
/// The exactness derivation: the predicate promotes the stored `f32`
/// coordinates losslessly, so each assertion is evaluated exactly and
/// carries no numeric tolerance. The only `f32` error surface is the
/// source-to-stored rounding, and its bound here is zero: the fixture is
/// dyadic, so `Scalar::from_f64` rounds nothing and the stored
/// configuration is the source configuration. A `Degenerate` (exactly
/// on-sphere) result would mean a co-spherical five-point subset,
/// falsifying the general-position premise, so every non-corner vertex
/// must assert strictly-outside.
#[test]
fn f32_indexed_mesh_tetrahedralization_is_strictly_delaunay() {
    let source = dyadic_points();
    let points: Vec<Point3<f32>> = source
        .iter()
        .map(|p| {
            Point3::new(
                <f32 as Scalar>::from_f64(p.x),
                <f32 as Scalar>::from_f64(p.y),
                <f32 as Scalar>::from_f64(p.z),
            )
        })
        .collect();
    let (vertices, tets) = run_bowyer_watson(&points);

    let mut builder = TetrahedralMeshBuilder::<f32>::new();
    let ids: Vec<VertexId> = vertices
        .iter()
        .map(|v| builder.vertex_array([v.x, v.y, v.z]))
        .collect();
    for tet in &tets {
        builder
            .tetrahedron([ids[tet[0]], ids[tet[1]], ids[tet[2]], ids[tet[3]]])
            .expect("invariant: engine-finalized tets are valid builder cells");
    }
    let mesh = builder.build();

    assert_eq!(mesh.vertex_count(), source.len());
    assert_eq!(mesh.cell_count(), tets.len());

    let at = |q: &Point3<f32>| [q.x, q.y, q.z];
    for cell in &mesh.cells {
        let mut corners: Vec<&Point3<f32>> = cell
            .vertex_ids
            .iter()
            .map(|&i| mesh.vertices.position(VertexId::from_usize(i)))
            .collect();
        // The builder canonicalizes every cell to right-hand-rule
        // positive orientation — the Shewchuk-negative convention —
        // while `insphere` requires Shewchuk-positive, so apply the
        // kernel's own swap rule (`Tetrahedron::new`) first.
        if orient_3d(
            at(corners[0]),
            at(corners[1]),
            at(corners[2]),
            at(corners[3]),
        )
        .is_positive()
        {
            corners.swap(2, 3);
        }
        for (vertex_id, _) in mesh.vertices.iter() {
            if cell.vertex_ids.contains(&vertex_id.as_usize()) {
                continue;
            }
            let orientation = insphere(
                at(corners[0]),
                at(corners[1]),
                at(corners[2]),
                at(corners[3]),
                at(mesh.vertices.position(vertex_id)),
            );
            assert_eq!(
                orientation,
                Orientation::Negative,
                "vertex {vertex_id:?} must lie strictly outside the circumsphere of cell {:?}",
                cell.vertex_ids
            );
        }
    }
}

/// GAIA-002: on dyadic inputs both precisions store identical values and
/// evaluate the exact predicates on them. The Delaunay tetrahedralization
/// of a point set in general position is unique, and the super-tetrahedron
/// anchors are discarded before `finalize`, so the `f32` and `f64` runs
/// must produce the identical tet set — the native `f32` path is the same
/// computation, not a parallel one.
#[test]
fn f32_and_f64_tetrahedralizations_agree_on_dyadic_inputs() {
    let source = dyadic_points();
    let source32: Vec<Point3<f32>> = source
        .iter()
        .map(|p| {
            Point3::new(
                <f32 as Scalar>::from_f64(p.x),
                <f32 as Scalar>::from_f64(p.y),
                <f32 as Scalar>::from_f64(p.z),
            )
        })
        .collect();
    let (_, tets32) = run_bowyer_watson(&source32);
    let (_, tets64) = run_bowyer_watson(&source);

    let canonical = |tets: &[[usize; 4]]| {
        let mut keys: Vec<[usize; 4]> = tets
            .iter()
            .map(|t| {
                let mut sorted = *t;
                sorted.sort_unstable();
                sorted
            })
            .collect();
        keys.sort_unstable();
        keys
    };

    assert_eq!(canonical(&tets32), canonical(&tets64));
}
