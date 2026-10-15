fn old_numeric<T: eunomia::RealField>(value: T) -> T {
    value
}

fn inherited_contract<T: gaia::domain::core::Scalar>(value: T) -> T {
    old_numeric(value)
}

fn main() {
    assert_eq!(inherited_contract(2.0_f64), 2.0);
    assert_eq!(inherited_contract(2.0_f32), 2.0);
}
