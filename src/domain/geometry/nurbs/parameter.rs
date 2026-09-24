use crate::domain::core::scalar::Scalar;

/// Interpolate a uniform sample after converting the division to the scalar type.
#[inline]
pub(super) fn uniform_parameter<T: Scalar>(low: T, high: T, index: usize, segments: usize) -> T {
    let fraction = <T as Scalar>::from_f64(index as f64) / <T as Scalar>::from_f64(segments as f64);
    low + (high - low) * fraction
}

#[cfg(test)]
mod tests {
    use super::uniform_parameter;
    use crate::domain::core::scalar::Scalar;

    #[test]
    fn uniform_parameter_divides_in_native_scalar_precision() {
        let segments = 16_777_217_usize;
        let actual = uniform_parameter(0.0_f32, 1.0_f32, 1, segments);
        let widened = <f32 as Scalar>::from_f64(1.0 / segments as f64);

        assert_eq!(actual.to_bits(), 0x3380_0000);
        assert_eq!(widened.to_bits(), 0x337f_ffff);
        assert_ne!(actual.to_bits(), widened.to_bits());
    }
}
