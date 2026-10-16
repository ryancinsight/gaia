//! Manual public-API driver for exact midpoint and differential bit checks.
use gaia::domain::{core::scalar::Scalar, geometry::Aabb};
use leto::geometry::Point3;
use std::io::{self, BufRead};

fn observe<T: Scalar>(index: usize, values: [T; 6], bits: impl Fn(T) -> u64) {
    let bounds = Aabb::new(
        Point3::new(values[0], values[1], values[2]),
        Point3::new(values[3], values[4], values[5]),
    );
    let center = bounds.center();
    println!(
        "{index} {} {} {}",
        bits(center.x),
        bits(center.y),
        bits(center.z)
    );
}

fn main() {
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line = line.expect("fixture line");
        let mut fields = line.split_whitespace();
        let width = fields.next().expect("precision");
        let values: Vec<_> = fields
            .map(|field| u64::from_str_radix(field, 16).expect("IEEE bits"))
            .collect();
        assert_eq!(values.len(), 6);
        match width {
            "32" => observe(
                index,
                std::array::from_fn(|i| {
                    f32::from_bits(u32::try_from(values[i]).expect("binary32 bits"))
                }),
                |v: f32| u64::from(v.to_bits()),
            ),
            "64" => observe(
                index,
                std::array::from_fn(|i| f64::from_bits(values[i])),
                f64::to_bits,
            ),
            _ => panic!("supported precision"),
        }
    }
}
