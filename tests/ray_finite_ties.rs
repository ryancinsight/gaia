//! Finite slab intervals must intersect before their rounded distances tie.

use eunomia::NumericElement;
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::{Aabb, Ray};
use leto::geometry::{Point3, Vector3};

fn point<T: Scalar>(p: [T; 3]) -> Point3<T> {
    Point3::new(p[0], p[1], p[2])
}

fn reject_disjoint_slabs<T: Scalar>(x: [T; 2], y: [T; 2]) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    // The ray has equal stored direction magnitudes. Hence |x(t)|=|y(t)|
    // for every real t, whereas these two intervals have a strict gap.
    assert!(x[1] < y[0] || y[1] < x[0]);
    for sx in [-one, one] {
        for sy in [-one, one] {
            for rotation in 0..3 {
                let rotate = |mut p: [T; 3]| {
                    p.rotate_left(rotation);
                    p
                };
                let d = rotate([sx, sy, zero]);
                let ray = Ray::try_new(point([zero; 3]), Vector3::new(d[0], d[1], d[2]))
                    .expect("finite nonzero direction");
                let stored = ray.direction();
                // Rotation preserves the multiset; compare the nonzero components.
                let mut magnitudes = [stored.x.abs(), stored.y.abs(), stored.z.abs()];
                magnitudes.sort_by(|a, b| a.partial_cmp(b).expect("finite direction"));
                assert_eq!(magnitudes[0], zero);
                assert_eq!(magnitudes[1], magnitudes[2]);
                let lower = rotate([
                    (sx * x[0]).min_scalar(sx * x[1]),
                    (sy * y[0]).min_scalar(sy * y[1]),
                    -one,
                ]);
                let upper = rotate([
                    (sx * x[0]).max_scalar(sx * x[1]),
                    (sy * y[0]).max_scalar(sy * y[1]),
                    one,
                ]);
                assert_eq!(
                    ray.intersect_aabb(&Aabb::new(point(lower), point(upper))),
                    None,
                    "rotation={rotation}, signs=({sx:?},{sy:?})"
                );
            }
        }
    }
}

fn retain_contact<T: Scalar>(expected_distance: T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let half = <T as Scalar>::from_f64(0.5);
    let ray = Ray::try_new(point([zero; 3]), Vector3::new(one, one, zero))
        .expect("finite diagonal direction");
    // The x interval ends exactly where the y interval begins: a real contact.
    let bounds = Aabb::new(point([zero, half, -one]), point([half, one, one]));
    assert_eq!(
        ray.intersect_aabb(&bounds),
        Some((expected_distance, expected_distance))
    );
}

fn retain_certified_contacts<T: Scalar>() {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let two = <T as Scalar>::from_int(2);
    for sign in [-one, one] {
        for rotation in 0..3 {
            let rotate = |mut p: [T; 3]| {
                p.rotate_left(rotation);
                p
            };
            let origin = point(rotate([sign * two, -one, one]));
            let d = rotate([sign, zero, zero]);
            let ray = Ray::try_new(origin, Vector3::new(d[0], d[1], d[2]))
                .expect("finite axis direction");
            let target = origin + ray.direction();
            assert_eq!(
                ray.intersect_aabb(&Aabb::new(target, target)),
                Some((one, one))
            );

            let ray = Ray::try_new(origin, Vector3::new(sign, -one, one))
                .expect("finite oblique direction");
            assert_eq!(
                ray.intersect_aabb(&Aabb::new(origin, origin)),
                Some((zero, zero))
            );
        }
    }
}

fn retain_tiny_origin_contact<T: Scalar>(tiny: T, next_half: T, expected_distance: T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let half = <T as Scalar>::from_f64(0.5);
    let ray = Ray::try_new(point([zero, tiny, zero]), Vector3::new(one, one, zero))
        .expect("finite diagonal direction");
    // At x=0.5 the exact stored ray has y=0.5+tiny, inside [0.5,next(0.5)].
    // Native subtraction loses tiny, but exact ordering must retain the contact.
    // This mixed-exponent contact can require allocating predicate work; the
    // cheap shared-origin contacts do not bound its latency or allocation cost.
    let bounds = Aabb::new(point([half, half, -one]), point([half, next_half, one]));
    assert_eq!(
        ray.intersect_aabb(&bounds),
        Some((expected_distance, expected_distance))
    );
}

#[test]
fn disjoint_point_slabs_binary32() {
    let x = f32::from_bits(0x3eff_ffff);
    let y = f32::from_bits(0x3eff_fffe);
    reject_disjoint_slabs([x; 2], [y; 2]);
}

#[test]
fn disjoint_positive_width_slabs_binary32() {
    reject_disjoint_slabs(
        [f32::from_bits(0x3eff_fffd), f32::from_bits(0x3eff_fffe)],
        [f32::from_bits(0x3eff_ffff), 0.5],
    );
}

#[test]
fn disjoint_point_slabs_binary64() {
    let x = f64::from_bits(0x3fdf_ffff_ffff_ffff);
    let y = f64::from_bits(0x3fdf_ffff_ffff_fffe);
    reject_disjoint_slabs([x; 2], [y; 2]);
}

#[test]
fn disjoint_positive_width_slabs_binary64() {
    reject_disjoint_slabs(
        [
            f64::from_bits(0x3fdf_ffff_ffff_fffd),
            f64::from_bits(0x3fdf_ffff_ffff_fffe),
        ],
        [f64::from_bits(0x3fdf_ffff_ffff_ffff), 0.5],
    );
}

#[test]
fn genuine_contact_binary32() {
    retain_contact(f32::from_bits(0x3f35_04f3));
}

#[test]
fn genuine_contact_binary64() {
    retain_contact(f64::from_bits(0x3fe6_a09e_667f_3bcd));
}

#[test]
fn origin_and_cardinal_contacts_binary32() {
    retain_certified_contacts::<f32>();
}

#[test]
fn origin_and_cardinal_contacts_binary64() {
    retain_certified_contacts::<f64>();
}

#[test]
fn mixed_exponent_finite_contact_binary32() {
    retain_tiny_origin_contact(
        f32::from_bits(1),
        f32::from_bits(0x3f00_0001),
        f32::from_bits(0x3f35_04f3),
    );
}

#[test]
fn mixed_exponent_finite_contact_binary64() {
    retain_tiny_origin_contact(
        f64::from_bits(1),
        f64::from_bits(0x3fe0_0000_0000_0001),
        f64::from_bits(0x3fe6_a09e_667f_3bcd),
    );
}
