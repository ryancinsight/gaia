//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::{Point3r, Vector3r};

#[test]
fn cdt_fill_loop_triangulates_concave_polygon() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let poly = vec![
        pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(2.0, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(2.0, 1.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(1.0, 1.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(1.0, 2.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(0.0, 2.0, 0.0), n),
    ];

    let mut out = Vec::new();
    let mut valence = HashMap::new();
    let added = cdt_fill_loop(&poly, &pool, &mut out, &mut valence);

    assert_eq!(
        added,
        poly.len() - 2,
        "simple concave polygon should triangulate to n-2 triangles"
    );

    let mut area_sum = 0.0_f64;
    for f in &out {
        let a = pool.position(f.vertices[0]);
        let b = pool.position(f.vertices[1]);
        let c = pool.position(f.vertices[2]);
        let area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        assert!(area2.abs() > 1e-12, "CDT fill produced degenerate triangle");
        area_sum += area2.abs() * 0.5;
    }
    assert!(
        (area_sum - 3.0).abs() < 1e-9,
        "triangulated area must match polygon area (got {area_sum:.12})"
    );
}

#[test]
fn cdt_fill_loop_rejects_collinear_polygon() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let poly = vec![
        pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(2.0, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(3.0, 0.0, 0.0), n),
    ];

    let mut out = Vec::new();
    let mut valence = HashMap::new();
    let added = cdt_fill_loop(&poly, &pool, &mut out, &mut valence);

    assert_eq!(added, 0, "collinear loop must not be triangulated");
    assert!(
        out.is_empty(),
        "no faces should be emitted for collinear loop"
    );
}
