//! Exact miss-contract regressions when native slab endpoints overflow.

use eunomia::NumericElement;
use gaia::domain::{
    core::scalar::Scalar,
    geometry::{Aabb, Ray},
};
use leto::geometry::{Point3, Vector3};

fn unit_ray<T: Scalar>() -> Ray<T> {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let ray = Ray::try_new(Point3::new(zero, zero, zero), Vector3::new(one, one, zero))
        .expect("finite nonzero analytical ray direction");
    assert!(ray.direction().x > zero && ray.direction().x == ray.direction().y);
    ray
}

fn overflow_miss<T: Scalar>(maximum: T, previous: T) -> Option<(T, T)> {
    let ray = unit_ray::<T>();
    assert_eq!(ray.direction().x, ray.direction().y);
    assert!(previous < maximum);
    // Every real point on this ray has x=y, while this finite point has x!=y.
    let point = Point3::new(maximum, previous, <T as NumericElement>::ZERO);
    ray.intersect_aabb(&Aabb::new(point, point))
}

fn overflow_interval_miss<T: Scalar>(maximum: T, previous: T) -> Option<(T, T)> {
    let one = <T as NumericElement>::ONE;
    let gap = maximum - previous;
    let x_hi = previous - gap;
    let x_lo = x_hi - gap;
    // Both x/y intervals have positive width, but every x is below every y.
    assert!(x_lo < x_hi && x_hi < previous && previous < maximum);
    let bounds = Aabb::new(
        Point3::new(x_lo, previous, -one),
        Point3::new(x_hi, maximum, one),
    );
    let ray = unit_ray::<T>();
    assert_eq!(ray.direction().x, ray.direction().y);
    ray.intersect_aabb(&bounds)
}

#[test]
fn out_of_range_distances_do_not_turn_distinct_coordinates_into_a_hit_single() {
    assert_eq!(overflow_miss(f32::MAX, f32::from_bits(0x7f7f_fffe)), None);
}
#[test]
fn out_of_range_distances_do_not_turn_distinct_coordinates_into_a_hit_double() {
    assert_eq!(
        overflow_miss(f64::MAX, f64::from_bits(0x7fef_ffff_ffff_fffe)),
        None
    );
}

#[test]
fn positive_width_intervals_cannot_intersect_when_all_x_are_below_all_y_single() {
    assert_eq!(
        overflow_interval_miss(f32::MAX, f32::from_bits(0x7f7f_fffe)),
        None
    );
}
#[test]
fn positive_width_intervals_cannot_intersect_when_all_x_are_below_all_y_double() {
    assert_eq!(
        overflow_interval_miss(f64::MAX, f64::from_bits(0x7fef_ffff_ffff_fffe)),
        None
    );
}

fn assert_true_hits<T: Scalar>(maximum: T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let infinity = <T as eunomia::RealField>::infinity();
    for signs in [[one, one], [-one, one], [one, -one], [-one, -one]] {
        for rotation in 0..3 {
            let rotate = |mut p: [T; 3]| {
                p.rotate_left(rotation);
                p
            };
            let direction = rotate([signs[0], signs[1], zero]);
            let target = rotate([signs[0] * maximum, signs[1] * maximum, zero]);
            let target = Point3::new(target[0], target[1], target[2]);
            let ray = Ray::try_new(
                Point3::new(zero, zero, zero),
                Vector3::new(direction[0], direction[1], direction[2]),
            )
            .expect("finite nonzero analytical ray direction");
            assert_eq!(
                ray.intersect_aabb(&Aabb::new(target, target)),
                Some((infinity, infinity))
            );
            let far = Ray::try_new(
                target,
                Vector3::new(-direction[0], -direction[1], -direction[2]),
            )
            .expect("finite nonzero analytical ray direction");
            let opposite = Point3::new(-target.x, -target.y, -target.z);
            assert_eq!(
                far.intersect_aabb(&Aabb::new(opposite, opposite)),
                Some((infinity, infinity))
            );
        }
    }
}
#[test]
fn true_unrepresentable_hits_remain_hits_single() {
    assert_true_hits(f32::MAX);
}
#[test]
fn true_unrepresentable_hits_remain_hits_double() {
    assert_true_hits(f64::MAX);
}
