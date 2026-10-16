use gaia::domain::geometry::predicates::{self as geometry, Orientation};
use gaia::domain::topology::predicates::{self as topology, Sign};
use leto::geometry::{Point2, Point3};

#[derive(Debug)]
struct Observation {
    operation: &'static str,
    fixture: String,
    expected: Orientation,
    actual: Orientation,
}

fn sign(value: Sign) -> Orientation {
    match value {
        Sign::Positive => Orientation::Positive,
        Sign::Negative => Orientation::Negative,
        Sign::Zero => Orientation::Degenerate,
    }
}

fn observations() -> Vec<Observation> {
    let mut rows = Vec::new();
    let mut add = |operation, fixture, expected, actual| {
        rows.push(Observation {
            operation,
            fixture,
            expected,
            actual,
        });
    };
    // Positive powers of two are represented exactly. These determinants are
    // monomials: scale^2, scale^3, 2*scale^4 and 2*scale^5, respectively. No approximate oracle.
    for (name, exponent) in [
        ("unit", 0),
        ("orient_overflow", 600),
        ("orient_underflow", -600),
        ("volume_overflow", 400),
        ("volume_underflow", -400),
        ("circle_overflow", 300),
        ("circle_underflow", -300),
        ("sphere_overflow", 240),
        ("sphere_underflow", -240),
        ("f32_min_subnormal", -149),
        ("f32_large_finite", 126),
    ] {
        let scale = 2.0_f64.powi(exponent);
        assert!(scale.is_finite() && scale > 0.0);
        let first = [0.0, 0.0];
        let second = [scale, 0.0];
        let third = [0.0, scale];
        for (order, second, third, expected) in [
            ("forward", second, third, Orientation::Positive),
            ("reverse", third, second, Orientation::Negative),
        ] {
            let fixture = format!("{name}:{order}:2^{exponent}");
            add(
                "geometry.orient_2d_arr",
                fixture.clone(),
                expected,
                geometry::orient_2d_arr(first, second, third),
            );
            add(
                "geometry.orient_2d",
                fixture.clone(),
                expected,
                geometry::orient_2d(
                    &Point2::from(first),
                    &Point2::from(second),
                    &Point2::from(third),
                ),
            );
            let p = |v: [f64; 2]| Point3::new(v[0], v[1], 0.0);
            add(
                "topology.orient2d",
                fixture,
                expected,
                sign(topology::orient2d(&p(first), &p(second), &p(third))),
            );
        }
        let origin = [0.0; 3];
        let x_axis = [scale, 0.0, 0.0];
        let y_axis = [0.0, scale, 0.0];
        let z_axis = [0.0, 0.0, scale];
        for (order, x_axis, y_axis, expected, opposite) in [
            (
                "forward",
                x_axis,
                y_axis,
                Orientation::Positive,
                Orientation::Negative,
            ),
            (
                "reverse",
                y_axis,
                x_axis,
                Orientation::Negative,
                Orientation::Positive,
            ),
        ] {
            let fixture = format!("{name}:{order}:2^{exponent}");
            add(
                "geometry.orient_3d",
                fixture.clone(),
                expected,
                geometry::orient_3d(origin, x_axis, y_axis, z_axis),
            );
            add(
                "geometry.orient_3d_pts",
                fixture.clone(),
                expected,
                geometry::orient_3d_pts(
                    &Point3::from(origin),
                    &Point3::from(x_axis),
                    &Point3::from(y_axis),
                    &Point3::from(z_axis),
                ),
            );
            add(
                "topology.orient3d",
                fixture,
                opposite,
                sign(topology::orient3d(
                    &Point3::from(origin),
                    &Point3::from(x_axis),
                    &Point3::from(y_axis),
                    &Point3::from(z_axis),
                )),
            );
        }
        let first = [scale, 0.0];
        let second = [0.0, scale];
        let third = [-scale, 0.0];
        let fourth = [0.0; 2];
        for (order, first, second, expected) in [
            ("forward", first, second, Orientation::Positive),
            ("reverse", second, first, Orientation::Negative),
        ] {
            let fixture = format!("{name}:{order}:2^{exponent}");
            add(
                "geometry.incircle",
                fixture.clone(),
                expected,
                geometry::incircle(
                    &Point2::from(first),
                    &Point2::from(second),
                    &Point2::from(third),
                    &Point2::from(fourth),
                ),
            );
            let p = |v: [f64; 2]| Point3::new(v[0], v[1], 0.0);
            add(
                "topology.incircle2d",
                fixture,
                expected,
                sign(topology::incircle2d(
                    &p(first),
                    &p(second),
                    &p(third),
                    &p(fourth),
                )),
            );
        }
        let first = [scale, 0.0, 0.0];
        let second = [0.0, scale, 0.0];
        let third = [0.0, 0.0, scale];
        let fourth = [-scale, 0.0, 0.0];
        let query = [0.0; 3];
        for (order, first, second, expected) in [
            ("forward", first, second, Orientation::Positive),
            ("reverse", second, first, Orientation::Negative),
        ] {
            let fixture = format!("{name}:{order}:2^{exponent}");
            add(
                "geometry.insphere",
                fixture.clone(),
                expected,
                geometry::insphere(first, second, third, fourth, query),
            );
            add(
                "topology.insphere3d",
                fixture,
                expected,
                sign(topology::insphere3d(
                    &Point3::from(first),
                    &Point3::from(second),
                    &Point3::from(third),
                    &Point3::from(fourth),
                    &Point3::from(query),
                )),
            );
        }
    }
    // This determinant is 2^600 even though the provider'scale intermediate
    // products overflow for particular permutations. Even permutations preserve
    // sign; odd permutations negate it. Every stored coordinate is finite.
    let large = 2.0_f64.powi(600);
    let points = [[0.0, 0.0], [1.0, 0.0], [large, large]];
    for (order, expected) in [
        ([0, 1, 2], Orientation::Positive),
        ([1, 2, 0], Orientation::Positive),
        ([2, 0, 1], Orientation::Positive),
        ([1, 0, 2], Orientation::Negative),
        ([2, 1, 0], Orientation::Negative),
        ([0, 2, 1], Orientation::Negative),
    ] {
        let [first, second, third] = order.map(|i| points[i]);
        add(
            "geometry.orient_2d_arr",
            format!("finite_determinant_overflow:{order:?}"),
            expected,
            geometry::orient_2d_arr(first, second, third),
        );
    }
    for (name, scale) in [
        ("actual_f32_unit", 1.0_f32),
        ("actual_f32_min_subnormal", f32::from_bits(1)),
        ("actual_f32_large_finite", f32::from_bits(253_u32 << 23)),
    ] {
        let origin = [0.0_f32; 3];
        let x_axis = [scale, 0.0, 0.0];
        let y_axis = [0.0, scale, 0.0];
        let z_axis = [0.0, 0.0, scale];
        for (order, x_axis, y_axis, expected) in [
            ("forward", x_axis, y_axis, Orientation::Positive),
            ("reverse", y_axis, x_axis, Orientation::Negative),
        ] {
            let fixture = format!("{name}:{order}");
            add(
                "geometry.orient_2d_arr",
                fixture.clone(),
                expected,
                geometry::orient_2d_arr([0.0; 2], [x_axis[0], x_axis[1]], [y_axis[0], y_axis[1]]),
            );
            add(
                "geometry.orient_2d",
                fixture.clone(),
                expected,
                geometry::orient_2d(
                    &Point2::new(0.0, 0.0),
                    &Point2::new(x_axis[0], x_axis[1]),
                    &Point2::new(y_axis[0], y_axis[1]),
                ),
            );
            add(
                "geometry.orient_3d",
                fixture.clone(),
                expected,
                geometry::orient_3d(origin, x_axis, y_axis, z_axis),
            );
            add(
                "geometry.orient_3d_pts",
                fixture,
                expected,
                geometry::orient_3d_pts(
                    &Point3::from(origin),
                    &Point3::from(x_axis),
                    &Point3::from(y_axis),
                    &Point3::from(z_axis),
                ),
            );
        }
        for (order, first, second, expected) in [
            ("forward", [scale, 0.0], [0.0, scale], Orientation::Positive),
            ("reverse", [0.0, scale], [scale, 0.0], Orientation::Negative),
        ] {
            let fixture = format!("{name}:{order}");
            add(
                "geometry.incircle",
                fixture.clone(),
                expected,
                geometry::incircle(
                    &Point2::from(first),
                    &Point2::from(second),
                    &Point2::new(-scale, 0.0),
                    &Point2::new(0.0, 0.0),
                ),
            );
            add(
                "geometry.insphere",
                fixture,
                expected,
                geometry::insphere(
                    [first[0], first[1], 0.0],
                    [second[0], second[1], 0.0],
                    [0.0, 0.0, scale],
                    [-scale, 0.0, 0.0],
                    [0.0; 3],
                ),
            );
        }
    }
    rows
}

#[cfg(test)]
fn verify(operation: &str, fixture: &str) {
    let selected: Vec<_> = observations()
        .into_iter()
        .filter(|r| r.operation == operation && r.fixture.contains(fixture))
        .collect();
    assert!(!selected.is_empty());
    let failures: Vec<_> = selected
        .into_iter()
        .filter(|r| r.actual != r.expected)
        .collect();
    assert!(
        failures.is_empty(),
        "Exact analytical signs disagree: {failures:#?}"
    );
}

#[test]
fn orient2d_overflow() {
    verify("geometry.orient_2d_arr", "finite_determinant_overflow");
}
#[test]
fn orient2d_underflow() {
    verify("geometry.orient_2d_arr", "orient_underflow");
}
#[test]
fn orient3d_overflow() {
    verify("geometry.orient_3d", "volume_overflow");
}
#[test]
fn orient3d_underflow() {
    verify("geometry.orient_3d", "volume_underflow");
}
#[test]
fn incircle_overflow() {
    verify("geometry.incircle", "circle_overflow");
}
#[test]
fn incircle_underflow() {
    verify("geometry.incircle", "circle_underflow");
}
#[test]
fn insphere_overflow() {
    verify("geometry.insphere", "sphere_overflow");
}
#[test]
fn insphere_underflow() {
    verify("geometry.insphere", "sphere_underflow");
}
#[test]
fn topology_orient2d_underflow() {
    verify("topology.orient2d", "orient_underflow");
}
#[test]
fn topology_orient3d_underflow() {
    verify("topology.orient3d", "volume_underflow");
}
#[test]
fn topology_incircle_underflow() {
    verify("topology.incircle2d", "circle_underflow");
}
#[test]
fn topology_insphere_underflow() {
    verify("topology.insphere3d", "sphere_underflow");
}
#[test]
fn actual_f32_controls() {
    let rows: Vec<_> = observations()
        .into_iter()
        .filter(|r| r.fixture.starts_with("actual_f32"))
        .collect();
    assert_eq!(rows.len(), 36);
    for row in rows {
        assert_eq!(row.actual, row.expected, "{row:?}");
    }
}

#[cfg(not(test))]
fn main() {
    println!("operation\tfixture\texpected\tactual\tverdict");
    for row in observations() {
        println!(
            "{}\t{}\t{:?}\t{:?}\t{}",
            row.operation,
            row.fixture,
            row.expected,
            row.actual,
            if row.expected == row.actual {
                "PASS"
            } else {
                "DEFECT"
            }
        );
    }
}
