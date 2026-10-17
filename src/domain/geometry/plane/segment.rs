//! Native range recovery for a plane/segment intersection parameter.

use crate::domain::core::scalar::Scalar;
use eunomia::NumericElement;

// Called only after the unchanged parallel check and a nonfinite denominator.
// Finite da/db can overflow da-db only with opposite signs. Both magnitudes
// then exceed the normal/subnormal boundary, so multiplication by 1/2 is exact.
// The halved denominator is finite and nonzero; their ratio stays in [0,1].
#[cold]
#[inline(never)]
pub(super) fn overflow_parameter<T: Scalar>(da: T, db: T, denom: T, eps: T) -> T {
    if <T as NumericElement>::is_finite(da)
        && <T as NumericElement>::is_finite(db)
        && <T as NumericElement>::is_finite(eps)
        && eps >= <T as NumericElement>::ZERO
    {
        let half = T::from_int(1) / T::from_int(2);
        let scaled_a = da * half;
        scaled_a / (scaled_a - db * half)
    } else {
        // Preserve existing nonfinite-distance and invalid-epsilon arithmetic.
        da / denom
    }
}
