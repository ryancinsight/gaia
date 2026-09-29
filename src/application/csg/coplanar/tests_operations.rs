//! Tests for the parent module, extracted from the module body.

use super::super::geometry2d::polygon_area_2d;
use super::{aabb_overlaps, CoplanarBuffers, SweepAabbIndex2d, TriData};
use crate::application::csg::boolean::{csg_boolean, BooleanOp};
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::primitives::{Disk, PrimitiveMesh};
use crate::domain::mesh::IndexedMesh;

fn area(mesh: &IndexedMesh) -> f64 {
    mesh.faces
        .iter()
        .map(|f| {
            let a = mesh.vertices.position(f.vertices[0]);
            let b = mesh.vertices.position(f.vertices[1]);
            let c = mesh.vertices.position(f.vertices[2]);
            (b - a).cross(c - a).norm() * 0.5
        })
        .sum()
}

fn disk(cx: f64, r: f64, n: usize) -> IndexedMesh {
    use crate::domain::core::scalar::Point3r;
    Disk {
        center: Point3r::new(cx, 0., 0.),
        radius: r,
        segments: n,
    }
    .build()
    .unwrap()
}

#[test]
fn sweep_index_matches_bruteforce_candidates() {
    let opp_aabbs = vec![
        [-2.0, -1.0, -1.0, 1.0],
        [-0.5, -0.5, 0.5, 0.5],
        [0.25, -1.5, 1.0, -0.25],
        [1.0, 0.0, 2.0, 2.0],
        [-1.5, 1.1, -0.2, 2.2],
    ];
    let queries = vec![
        [-3.0, -0.2, -1.2, 0.2],
        [-0.75, -0.75, 0.75, 0.75],
        [0.1, -2.0, 1.1, -0.1],
        [0.8, -0.2, 1.8, 1.8],
        [-10.0, -10.0, 10.0, 10.0],
        [2.1, 2.1, 3.0, 3.0],
    ];

    let index = SweepAabbIndex2d::build(&opp_aabbs);
    let mut got = Vec::new();
    for q in queries {
        index.query_overlaps(&q, &opp_aabbs, &mut got);
        got.sort_unstable();

        let mut want: Vec<usize> = opp_aabbs
            .iter()
            .enumerate()
            .filter(|(_, b)| aabb_overlaps(&q, b))
            .map(|(i, _)| i)
            .collect();
        want.sort_unstable();

        assert_eq!(got, want, "query={q:?}");
    }
}

#[test]
fn sweep_index_orders_min_u_totally() {
    let opp_aabbs = vec![
        [0.0, -1.0, 1.0, 1.0],
        [f64::NAN, -1.0, 1.0, 1.0],
        [-0.0, -1.0, 1.0, 1.0],
        [f64::NEG_INFINITY, -1.0, 1.0, 1.0],
    ];
    let index = SweepAabbIndex2d::build(&opp_aabbs);
    let mut expected: Vec<usize> = (0..opp_aabbs.len()).collect();
    expected.sort_unstable_by(|&i, &j| opp_aabbs[i][0].total_cmp(&opp_aabbs[j][0]));

    assert_eq!(index.by_min_u, expected);
}

#[test]
fn coplanar_buffers_preserve_packed_triangle_order() {
    let data = vec![
        TriData {
            coords2d: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            aabb2d: [0.0, 0.0, 1.0, 1.0],
            verts3d: [
                Point3r::new(0.0, 0.0, 0.0),
                Point3r::new(1.0, 0.0, 0.0),
                Point3r::new(0.0, 1.0, 0.0),
            ],
        },
        TriData {
            coords2d: [1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
            aabb2d: [1.0, 1.0, 2.0, 2.0],
            verts3d: [
                Point3r::new(1.0, 1.0, 0.0),
                Point3r::new(2.0, 1.0, 0.0),
                Point3r::new(1.0, 2.0, 0.0),
            ],
        },
    ];

    let buffers = CoplanarBuffers::from_tri_data(&data);

    assert_eq!(
        buffers.tris(),
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
        ]
    );
    assert_eq!(
        buffers.aabbs(),
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
    );
}

#[test]
fn split_outside_partition_identity() {
    use crate::application::csg::clip::{clip_polygon_to_triangle, split_polygon_outside_triangle};

    let poly = vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
    let (dx, dy, ex, ey, fx, fy) = (0.5, 0.0, 1.5, 0.0, 1.0, 1.0);
    let a_poly = polygon_area_2d(&poly);
    let a_inside = polygon_area_2d(&clip_polygon_to_triangle(&poly, dx, dy, ex, ey, fx, fy));
    let a_outside: f64 = split_polygon_outside_triangle(&poly, dx, dy, ex, ey, fx, fy)
        .iter()
        .map(|p| polygon_area_2d(p))
        .sum();
    let err1 = ((a_inside + a_outside) - a_poly).abs();
    assert!(err1 < 1e-12, "case 1: partition err={err1:.2e}");
}

#[test]
fn identical_disks_union_equals_single() {
    let a = disk(0., 1., 64);
    let b = disk(0., 1., 64);
    let u = csg_boolean(BooleanOp::Union, &a, &b).unwrap();
    let err = (area(&u) - std::f64::consts::PI).abs() / std::f64::consts::PI;
    assert!(err < 0.02, "union err {:.1}%", err * 100.);
}

#[test]
fn offset_disks_intersection_area() {
    let (r, d) = (1.0_f64, 1.0_f64);
    let a = disk(-d / 2., r, 128);
    let b = disk(d / 2., r, 128);
    let inter = csg_boolean(BooleanOp::Intersection, &a, &b).unwrap();
    let th = (d / (2. * r)).acos();
    let exp = 2. * r * r * (th - th.sin() * th.cos());
    let err = (area(&inter) - exp).abs() / exp;
    assert!(err < 0.02, "inter err {:.1}%", err * 100.);
}

#[test]
fn disk_inclusion_exclusion() {
    let (r, d) = (1.0_f64, 1.0_f64);
    let a = disk(-d / 2., r, 128);
    let b = disk(d / 2., r, 128);
    let u = csg_boolean(BooleanOp::Union, &a, &b).unwrap();
    let i = csg_boolean(BooleanOp::Intersection, &a, &b).unwrap();
    let lhs = area(&a) + area(&b);
    let rhs = area(&u) + area(&i);
    let err = (lhs - rhs).abs() / lhs;
    assert!(err < 0.02, "incl-excl err {:.1}%", err * 100.);
}

#[test]
fn disk_difference_area() {
    let (r, d) = (1.0_f64, 1.0_f64);
    let a = disk(-d / 2., r, 128);
    let b = disk(d / 2., r, 128);
    let diff = csg_boolean(BooleanOp::Difference, &a, &b).unwrap();
    let inter = csg_boolean(BooleanOp::Intersection, &a, &b).unwrap();
    let area_a = area(&a);
    let lhs = area(&diff) + area(&inter);
    let err = (lhs - area_a).abs() / area_a;
    assert!(err < 0.05, "inclusion-exclusion err {:.1}%", err * 100.);
}
