//! Scalar type abstraction — zero-cost generic floating-point precision.
//!
//! # Design
//!
//! Instead of a compile-time feature flag that forces a single precision
//! across the whole crate, every mesh type is generic over `T: Scalar`.
//! Monomorphisation generates optimal machine code per instantiation —
//! identical to a hand-written `f64`-only implementation — while letting
//! callers freely mix `IndexedMesh<f32>` and `IndexedMesh<f64>` in one
//! binary without recompilation.
//!
//! # Theorem: Scalar completeness
//!
//! **Statement**: `Scalar` covers exactly the floating-point types that
//! support all mesh-geometry operations required by `gaia`.
//!
//! **Proof sketch**: `eunomia::RealField` provides a complete ordered field
//! with the algebraic/trigonometric operations needed for vector maths over the
//! `leto` geometry types, plus `infinity`/`neg_infinity`/`floor`/`sqrt`/
//! `min_scalar`/`max_scalar` and lossless `to_f64` conversion for human-readable
//! outputs.  The sealed super-trait restricts the impl set to `{f32, f64}`,
//! matching IEEE 754 hardware support.

use leto::geometry::{Point3, Vector3};

// ── Sealed trait ──────────────────────────────────────────────────────────────
// Prevents downstream crates from implementing `Scalar` for arbitrary types.
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Zero-cost generic floating-point scalar for all CFD mesh operations.
///
/// Implemented only by `f32` and `f64`.  Mesh types parameterised by
/// `T: Scalar` monomorphise to zero-overhead code, equivalent to writing a
/// separate `f32` and `f64` implementation by hand.
///
/// # Choosing between precisions
///
/// | Precision | Tolerance | Use case |
/// |-----------|-----------|----------|
/// | `f64` (default) | 1 nm | High-fidelity CFD, validation, export to `OpenFOAM` |
/// | `f32` | 10 µm | GPU-side geometry staging where bandwidth matters |
///
/// # Example
///
/// ```rust
/// use gaia::IndexedMesh;
///
/// // Both coexist in the same binary — no feature flag, no recompilation:
/// let hi: IndexedMesh<f64> = IndexedMesh::new();
/// let lo: IndexedMesh<f32> = IndexedMesh::new();
/// assert_eq!(hi.vertex_count(), 0);
/// assert_eq!(lo.vertex_count(), 0);
/// ```
pub trait Scalar:
    eunomia::RealField
    + Copy
    + Default
    + std::fmt::Debug
    + std::fmt::Display
    + Send
    + Sync
    + 'static
    + private::Sealed
{
    /// Absolute geometry tolerance appropriate for this precision.
    ///
    /// - `f64` → `1 × 10⁻⁹` m  (sub-nanometer; millifluidic mm-scale geometry)
    /// - `f32` → `1 × 10⁻⁵` m  (10 µm; bounded by single-precision rounding)
    fn tolerance() -> Self;

    /// Convert an `f64` literal to this scalar type.
    ///
    /// Zero-cost identity for `f64`; one `as` cast for `f32`.
    /// Enables generic code to write `T::from_f64(0.5)` instead of `0.5_T`.
    fn from_f64(v: f64) -> Self;

    /// Convert a `usize` index to this scalar type.
    ///
    /// The canonical replacement for `i as T` in generic mesh code.
    /// Correct for indices up to 2^53 (the `f64` mantissa limit) — mesh element
    /// counts are well within this bound in practice.  Delegates to `from_f64`
    /// so generic code that works in either `f32` or `f64` automatically selects
    /// the right precision-correct construction path provided by eunomia's
    /// `FloatElement::from_f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gaia::domain::core::scalar::Scalar;
    ///
    /// assert_eq!(f64::from_usize(7), 7.0_f64);
    /// assert_eq!(f32::from_usize(7), 7.0_f32);
    /// ```
    #[inline]
    #[must_use]
    fn from_usize(n: usize) -> Self {
        // `usize as f64` is exact for all values ≤ 2^53; mesh element counts
        // never approach that bound.  Converting through f64 is the eunomia-
        // sanctioned route: FloatElement::from_f64 is the crate's explicit
        // precision-correct widening seam.
        #[expect(
            clippy::cast_precision_loss,
            reason = "usize mesh counts stay below the 2^53 exact-in-f64 bound documented above"
        )]
        {
            <Self as Scalar>::from_f64(n as f64)
        }
    }

    /// Convert a signed index (e.g. an offset or delta) to this scalar type.
    ///
    /// The canonical replacement for `k as T` where `k: isize` or `k: i64`.
    /// Sign is preserved; the absolute value must not exceed 2^53.
    #[inline]
    #[must_use]
    fn from_index(k: i64) -> Self {
        #[expect(
            clippy::cast_precision_loss,
            reason = "signed mesh indices stay within the documented 2^53 exact-in-f64 conversion bound"
        )]
        {
            <Self as Scalar>::from_f64(k as f64)
        }
    }

    /// Compare values using the IEEE 754 total order, including signed zero and NaN.
    ///
    /// ```
    /// use gaia::domain::core::scalar::Scalar;
    ///
    /// assert_eq!(
    ///     Scalar::total_cmp(&-0.0_f64, &0.0_f64),
    ///     core::cmp::Ordering::Less
    /// );
    /// ```
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering;

    /// Squared tolerance — avoids `sqrt` in distance comparisons.
    #[inline]
    #[must_use]
    fn tolerance_sq() -> Self {
        let t = Self::tolerance();
        t * t
    }
}

impl Scalar for f64 {
    #[inline]
    fn tolerance() -> Self {
        1e-9
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        f64::total_cmp(self, other)
    }
}

impl Scalar for f32 {
    #[inline]
    fn tolerance() -> Self {
        1e-5_f32
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        #[expect(
            clippy::cast_precision_loss,
            reason = "f32 Scalar implementation: deliberately truncating f64 to f32 precision — f32 is the lower-precision scalar type by design"
        )]
        {
            v as f32
        }
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        f32::total_cmp(self, other)
    }
}

// ── Default-precision convenience aliases ─────────────────────────────────────

/// Default scalar precision — `f64` for sub-nanometer millifluidic accuracy.
///
/// All existing code that references `Real` continues to compile unchanged.
/// New code should prefer the generic `T: Scalar` pattern.
pub type Real = f64;

/// 3-D point at default (`f64`) precision.
pub type Point3r = Point3<Real>;

/// 3-D vector at default (`f64`) precision.
pub type Vector3r = Vector3<Real>;

/// Absolute geometry tolerance at default precision (1 nm).
pub const TOLERANCE: Real = 1e-9;

/// Squared tolerance at default precision — avoids `sqrt` in distance checks.
pub const TOLERANCE_SQ: Real = TOLERANCE * TOLERANCE;

// ── Generic sanitisation helpers ──────────────────────────────────────────────

/// Replace NaN or ±Inf with zero — generic over any `T: Scalar`.
#[inline]
pub fn sanitize<T: Scalar>(v: T) -> T {
    if <T as eunomia::NumericElement>::is_finite(v) {
        v
    } else {
        <T as eunomia::NumericElement>::ZERO
    }
}

/// Replace NaN / ±Inf components of a point with zero — generic.
#[inline]
pub fn sanitize_point<T: Scalar>(p: &Point3<T>) -> Point3<T> {
    Point3::new(sanitize(p.x), sanitize(p.y), sanitize(p.z))
}

/// Replace NaN / ±Inf components of a vector with zero — generic.
#[inline]
pub fn sanitize_vector<T: Scalar>(v: &Vector3<T>) -> Vector3<T> {
    Vector3::new(sanitize(v.x), sanitize(v.y), sanitize(v.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_ordering() {
        assert!(
            f32::tolerance() > f64::tolerance() as f32,
            "f32 tolerance must be coarser than f64"
        );
    }

    #[test]
    fn total_order_distinguishes_signed_zero() {
        fn assert_total_order<T: Scalar>() {
            let negative_zero = -<T as Scalar>::from_f64(0.0);
            let positive_zero = <T as Scalar>::from_f64(0.0);
            let infinity = <T as eunomia::RealField>::infinity();
            let nan = <T as eunomia::RealField>::nan();
            assert_eq!(
                <T as Scalar>::total_cmp(&negative_zero, &positive_zero),
                core::cmp::Ordering::Less
            );
            assert_eq!(
                <T as Scalar>::total_cmp(&infinity, &nan),
                core::cmp::Ordering::Less
            );
        }

        assert_total_order::<f32>();
        assert_total_order::<f64>();
    }

    #[test]
    fn from_f64_identity_f64() {
        assert_eq!(f64::from_f64(1.0_f64).to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn from_f64_cast_f32() {
        let v: f32 = f32::from_f64(0.5_f64);
        assert!((v - 0.5_f32).abs() < 1e-7, "f32 cast must be accurate");
    }

    #[test]
    fn sanitize_finite_passthrough() {
        assert_eq!(sanitize(1.5_f64).to_bits(), 1.5_f64.to_bits());
        assert_eq!(sanitize(1.5_f32).to_bits(), 1.5_f32.to_bits());
    }

    #[test]
    fn sanitize_nan_to_zero() {
        assert_eq!(sanitize(f64::NAN).to_bits(), 0.0_f64.to_bits());
        assert_eq!(sanitize(f32::NAN).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn sanitize_inf_to_zero() {
        assert_eq!(sanitize(f64::INFINITY).to_bits(), 0.0_f64.to_bits());
        assert_eq!(sanitize(f32::NEG_INFINITY).to_bits(), 0.0_f32.to_bits());
    }
}
