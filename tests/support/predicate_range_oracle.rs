//! Execute independent full-exponent fixtures through the actual public APIs.

use gaia::domain::geometry::predicates::{self as geometry, Orientation};
use gaia::domain::topology::predicates::{self as topology, Sign};
use leto::geometry::{Point2, Point3};
use std::io::{self, BufRead};

fn sign(value: Orientation) -> i64 {
    match value {
        Orientation::Positive => 1,
        Orientation::Negative => -1,
        Orientation::Degenerate => 0,
    }
}
fn topology_sign(value: Sign) -> i64 {
    match value {
        Sign::Positive => 1,
        Sign::Negative => -1,
        Sign::Zero => 0,
    }
}
fn two(value: &[f64]) -> [f64; 2] {
    [value[0], value[1]]
}
fn three(value: &[f64]) -> [f64; 3] {
    [value[0], value[1], value[2]]
}

fn main() {
    let mut cases = 0;
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let value: serde_json::Value =
            serde_json::from_str(&line.expect("fixture line")).expect("fixture JSON");
        let points: Vec<Vec<f64>> = value["points"]
            .as_array()
            .expect("points")
            .iter()
            .map(|point| {
                point
                    .as_array()
                    .expect("coordinates")
                    .iter()
                    .map(|bits| {
                        f64::from_bits(
                            u64::from_str_radix(bits.as_str().expect("hex bits"), 16)
                                .expect("binary64 bits"),
                        )
                    })
                    .collect()
            })
            .collect();
        assert!(points.iter().flatten().all(|value| value.is_finite()));
        let expected = value["expected"].as_i64().expect("exact sign");
        match value["kind"].as_str().expect("operation") {
            "orient2d" => {
                let [first, second, third] = std::array::from_fn(|row| two(&points[row]));
                assert_eq!(
                    sign(geometry::orient_2d_arr(first, second, third)),
                    expected,
                    "case {index}"
                );
                assert_eq!(
                    sign(geometry::orient_2d(
                        &Point2::from(first),
                        &Point2::from(second),
                        &Point2::from(third)
                    )),
                    expected,
                    "case {index}"
                );
                let point = |v: [f64; 2]| Point3::new(v[0], v[1], 0.0);
                assert_eq!(
                    topology_sign(topology::orient2d(
                        &point(first),
                        &point(second),
                        &point(third)
                    )),
                    expected,
                    "case {index}"
                );
            }
            "orient3d" => {
                let [first, second, third, fourth] = std::array::from_fn(|row| three(&points[row]));
                assert_eq!(
                    sign(geometry::orient_3d(first, second, third, fourth)),
                    expected,
                    "case {index}"
                );
                assert_eq!(
                    sign(geometry::orient_3d_pts(
                        &Point3::from(first),
                        &Point3::from(second),
                        &Point3::from(third),
                        &Point3::from(fourth)
                    )),
                    expected,
                    "case {index}"
                );
                assert_eq!(
                    topology_sign(topology::orient3d(
                        &Point3::from(first),
                        &Point3::from(second),
                        &Point3::from(third),
                        &Point3::from(fourth)
                    )),
                    -expected,
                    "case {index}"
                );
            }
            "incircle" => {
                let [first, second, third, query] =
                    std::array::from_fn(|row| Point2::from(two(&points[row])));
                assert_eq!(
                    sign(geometry::incircle(&first, &second, &third, &query)),
                    expected,
                    "case {index}"
                );
                let point = |v: Point2<f64>| Point3::new(v.x, v.y, 0.0);
                assert_eq!(
                    topology_sign(topology::incircle2d(
                        &point(first),
                        &point(second),
                        &point(third),
                        &point(query)
                    )),
                    expected,
                    "case {index}"
                );
            }
            "insphere" => {
                let [first, second, third, fourth, query] =
                    std::array::from_fn(|row| three(&points[row]));
                assert_eq!(
                    sign(geometry::insphere(first, second, third, fourth, query)),
                    expected,
                    "case {index}"
                );
                assert_eq!(
                    topology_sign(topology::insphere3d(
                        &Point3::from(first),
                        &Point3::from(second),
                        &Point3::from(third),
                        &Point3::from(fourth),
                        &Point3::from(query)
                    )),
                    expected,
                    "case {index}"
                );
            }
            other => panic!("unknown operation {other}"),
        }
        cases += 1;
    }
    println!("{cases} independent full-exponent cases pass through all public aliases");
}
