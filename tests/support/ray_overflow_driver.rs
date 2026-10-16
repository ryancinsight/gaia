//! Manual public-API driver for independent exact-rational ray/slab comparison.
use eunomia::NumericElement;
use gaia::domain::{core::scalar::Scalar, geometry::{Aabb, Ray}};
use leto::geometry::{Point3, Vector3};
use std::io::{self, BufRead};

fn observe<T: Scalar>(index: usize, values: [T; 12], bits: impl Fn(T) -> u64) {
    let point = |i| Point3::new(values[i], values[i + 1], values[i + 2]);
    let ray = Ray::try_new(point(0), Vector3::new(values[3], values[4], values[5]))
        .expect("finite nonzero fixture direction");
    let direction = ray.direction();
    assert!((0..3).all(|axis| <T as NumericElement>::is_finite(direction[axis])));
    let result = ray.intersect_aabb(&Aabb::new(point(6), point(9)));
    let (hit, entry, exit) = result.map_or((0, 0, 0), |(a, b)| (1, bits(a), bits(b)));
    println!("{index} {} {} {} {hit} {entry} {exit}", bits(direction.x), bits(direction.y), bits(direction.z));
}

fn main() {
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line = line.expect("fixture line");
        let mut fields = line.split_whitespace();
        let width = fields.next().expect("precision");
        let values: Vec<_> = fields.map(|field| u64::from_str_radix(field, 16).expect("IEEE bits")).collect();
        assert_eq!(values.len(), 12);
        match width {
            "32" => observe(index, std::array::from_fn(|i| f32::from_bits(u32::try_from(values[i]).expect("binary32 bits"))), |v: f32| u64::from(v.to_bits())),
            "64" => observe(index, std::array::from_fn(|i| f64::from_bits(values[i])), f64::to_bits),
            _ => panic!("supported precision"),
        }
    }
}
