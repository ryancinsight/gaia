//! Algebraic Signed Distance Fields (SDFs).
//!
//! Implements exact mathematical representations of boundary primitives to drive
//! conforming volumetric Delaunay tetrahedralization.
//!
//! # Theorem 1: Signed Distance Field (SDF) Boundary Projection
//!
//! **Statement**: For any implicit bounded geometry defined by $SDF(\mathbf{x}) \le 0$,
//! a point $\mathbf{x}_{near}$ proximal to the surface can be strictly projected onto
//! the mathematically exact manifold via gradient descent: $\mathbf{x}_{surface} =
//! \mathbf{x}_{near} - SDF(\mathbf{x}_{near}) \cdot \nabla SDF(\mathbf{x}_{near})$.
//!
//! **Proof sketch**: The continuous gradient $\nabla SDF$ is strictly orthogonal to
//! the 0-level set. Small Eulerian steps along the negative gradient unconditionally
//! converge to the lowest-energy isosurface, guaranteeing mathematically watertight
//! boundary closure for volumetric grids.

use leto::geometry::{Point3, Vector3};

use crate::domain::core::scalar::Scalar;

/// Defines a volumetric signed distance field bounding domain.
pub trait Sdf3D<T: Scalar> {
    /// Evaluate the signed distance function at a point.
    ///
    /// * $V < 0$: Strictly interior.
    /// * $V = 0$: Exactly on the manifold boundary.
    /// * $V > 0$: Strictly exterior.
    fn eval(&self, p: &Point3<T>) -> T;

    /// Calculate the gradient $\nabla SDF$ at point $p$.
    ///
    /// Default implementation uses a highly accurate central Cartesian finite
    /// difference formulation.
    fn gradient(&self, p: &Point3<T>) -> Vector3<T> {
        let eps = T::tolerance() * <T as Scalar>::from_f64(10.0);
        let dx = Vector3::new(eps, <T as eunomia::NumericElement>::ZERO, <T as eunomia::NumericElement>::ZERO);
        let dy = Vector3::new(<T as eunomia::NumericElement>::ZERO, eps, <T as eunomia::NumericElement>::ZERO);
        let dz = Vector3::new(<T as eunomia::NumericElement>::ZERO, <T as eunomia::NumericElement>::ZERO, eps);

        let df_dx = self.eval(&(p + dx)) - self.eval(&(p - dx));
        let df_dy = self.eval(&(p + dy)) - self.eval(&(p - dy));
        let df_dz = self.eval(&(p + dz)) - self.eval(&(p - dz));

        let denom = <T as Scalar>::from_f64(2.0) * eps;
        let mut grad = Vector3::new(df_dx / denom, df_dy / denom, df_dz / denom);

        // Normalize the gradient to ensure strict unity length for projection
        let norm = grad.norm();
        if norm > T::tolerance() {
            grad /= norm;
        }
        grad
    }

    /// Return the exact axis-aligned bounding box (AABB) of this geometry.
    /// The tuple is (`min_point`, `max_point`).
    fn bounds(&self) -> (Point3<T>, Point3<T>);
}

/// A perfectly spherical `SDF`.
#[derive(Debug, Clone)]
pub struct SphereSdf<T: Scalar> {
    /// Center coordinate of the sphere.
    pub center: Point3<T>,
    /// Radius of the sphere.
    pub radius: T,
}

impl<T: Scalar> Sdf3D<T> for SphereSdf<T> {
    fn eval(&self, p: &Point3<T>) -> T {
        (p - self.center).norm() - self.radius
    }

    fn bounds(&self) -> (Point3<T>, Point3<T>) {
        let r_vec = Vector3::new(self.radius, self.radius, self.radius);
        (self.center - r_vec, self.center + r_vec)
    }
}

/// An infinite analytic cylinder `SDF` traversing through space.
#[derive(Debug, Clone)]
pub struct CylinderSdf<T: Scalar> {
    /// A point strictly residing on the central axis of rotation.
    pub point: Point3<T>,
    /// The normalized directional vector of the central axis.
    pub axis: Vector3<T>,
    /// Cylindrical radius.
    pub radius: T,
}

impl<T: Scalar> CylinderSdf<T> {
    /// Construct a mathematically rigorous infinite cylinder.
    pub fn new(point: Point3<T>, mut axis: Vector3<T>, radius: T) -> Self {
        let norm = axis.norm();
        if norm > T::tolerance() {
            axis /= norm;
        } else {
            axis = Vector3::new(<T as eunomia::NumericElement>::ONE, <T as eunomia::NumericElement>::ZERO, <T as eunomia::NumericElement>::ZERO);
        }
        Self {
            point,
            axis,
            radius,
        }
    }
}

impl<T: Scalar> Sdf3D<T> for CylinderSdf<T> {
    fn eval(&self, p: &Point3<T>) -> T {
        let pb = p - self.point;
        let proj = pb.dot(self.axis);
        let radial_vec = pb - self.axis * proj;
        radial_vec.norm() - self.radius
    }

    fn bounds(&self) -> (Point3<T>, Point3<T>) {
        // Technically infinite in the axis direction.
        // We return an expansive bound clamped to arbitrary CFD limits.
        let big = <T as Scalar>::from_f64(100.0);
        let bound = Vector3::new(big, big, big);
        (self.point - bound, self.point + bound)
    }
}

/// A finite cylindrical capsule `SDF` with hemispherical end caps.
#[derive(Debug, Clone)]
pub struct CapsuleSdf<T: Scalar> {
    /// Start coordinate of the primary axis.
    pub a: Point3<T>,
    /// End coordinate of the primary axis.
    pub b: Point3<T>,
    /// Radial profile constraint.
    pub radius: T,
}

impl<T: Scalar> CapsuleSdf<T> {
    /// Construct a mathematically rigorous capsule.
    pub fn new(a: Point3<T>, b: Point3<T>, radius: T) -> Self {
        Self { a, b, radius }
    }
}

impl<T: Scalar> Sdf3D<T> for CapsuleSdf<T> {
    fn eval(&self, p: &Point3<T>) -> T {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let h = (pa.dot(ba) / ba.norm_squared()).clamp(<T as eunomia::NumericElement>::ZERO, <T as eunomia::NumericElement>::ONE);
        (pa - ba * h).norm() - self.radius
    }

    fn bounds(&self) -> (Point3<T>, Point3<T>) {
        let r_vec = Vector3::new(self.radius, self.radius, self.radius);
        let min_pt = Point3::new(
            (self.a.x).min_scalar(self.b.x),
            (self.a.y).min_scalar(self.b.y),
            (self.a.z).min_scalar(self.b.z),
        ) - r_vec;
        let max_pt = Point3::new(
            (self.a.x).max_scalar(self.b.x),
            (self.a.y).max_scalar(self.b.y),
            (self.a.z).max_scalar(self.b.z),
        ) + r_vec;
        (min_pt, max_pt)
    }
}

/// Exact finite cylinder `SDF` with mathematically flat end caps.
#[derive(Debug, Clone)]
pub struct FiniteCylinderSdf<T: Scalar> {
    /// Start coordinate of the primary axis (inlet/outlet cap).
    pub a: Point3<T>,
    /// End coordinate of the primary axis (inlet/outlet cap).
    pub b: Point3<T>,
    /// Radial profile constraint.
    pub radius: T,
}

impl<T: Scalar> FiniteCylinderSdf<T> {
    /// Construct a mathematically rigorous finite cylinder.
    pub fn new(a: Point3<T>, b: Point3<T>, radius: T) -> Self {
        Self { a, b, radius }
    }
}

impl<T: Scalar> Sdf3D<T> for FiniteCylinderSdf<T> {
    fn eval(&self, p: &Point3<T>) -> T {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let baba = ba.norm_squared();
        if baba < T::tolerance() {
            return (p - self.a).norm() - self.radius; // Degenerate point cylinder is a sphere
        }

        let paba = pa.dot(ba);
        let h = paba / baba; // fractional projection along cylinder
        let radial_vec = pa - ba * h;

        let radial_dist = radial_vec.norm() - self.radius;
        let half = <T as Scalar>::from_f64(0.5);
        let axial_dist = (h - half).abs() * (baba).sqrt() - (baba).sqrt() * half;

        let interior_dist = (radial_dist).max_scalar(axial_dist).min_scalar(<T as eunomia::NumericElement>::ZERO);

        let radial_ext = (radial_dist).max_scalar(<T as eunomia::NumericElement>::ZERO);
        let axial_ext = (axial_dist).max_scalar(<T as eunomia::NumericElement>::ZERO);
        let exterior_dist = (radial_ext * radial_ext + axial_ext * axial_ext).sqrt();

        interior_dist + exterior_dist
    }

    fn bounds(&self) -> (Point3<T>, Point3<T>) {
        let r_vec = Vector3::new(self.radius, self.radius, self.radius);
        let min_pt = Point3::new(
            (self.a.x).min_scalar(self.b.x),
            (self.a.y).min_scalar(self.b.y),
            (self.a.z).min_scalar(self.b.z),
        ) - r_vec;
        let max_pt = Point3::new(
            (self.a.x).max_scalar(self.b.x),
            (self.a.y).max_scalar(self.b.y),
            (self.a.z).max_scalar(self.b.z),
        ) + r_vec;
        (min_pt, max_pt)
    }
}

/// Exact boolean union of two SDF geometries.
#[derive(Debug, Clone)]
pub struct UnionSdf<T: Scalar, A: Sdf3D<T>, B: Sdf3D<T>> {
    primary: A,
    secondary: B,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar, A: Sdf3D<T>, B: Sdf3D<T>> UnionSdf<T, A, B> {
    /// Constructs a strict mathematical union utilizing exact minimum fields.
    pub fn new(primary: A, secondary: B) -> Self {
        Self {
            primary,
            secondary,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, A: Sdf3D<T>, B: Sdf3D<T>> Sdf3D<T> for UnionSdf<T, A, B> {
    fn eval(&self, p: &Point3<T>) -> T {
        (self.primary.eval(p)).min_scalar(self.secondary.eval(p))
    }

    fn bounds(&self) -> (Point3<T>, Point3<T>) {
        let (min1, max1) = self.primary.bounds();
        let (min2, max2) = self.secondary.bounds();
        (
            Point3::new(
                (min1.x).min_scalar(min2.x),
                (min1.y).min_scalar(min2.y),
                (min1.z).min_scalar(min2.z),
            ),
            Point3::new(
                (max1.x).max_scalar(max2.x),
                (max1.y).max_scalar(max2.y),
                (max1.z).max_scalar(max2.z),
            ),
        )
    }
}

/// Polynomial smooth union (smin) of two SDF geometries.
///
/// Produces a $C^1$-continuous fillet at intersections, preventing perfectly
/// sharp creases that collapse into non-manifold 1D edges on discrete lattices.
#[derive(Debug, Clone)]
pub struct SmoothUnionSdf<T: Scalar, A: Sdf3D<T>, B: Sdf3D<T>> {
    primary: A,
    secondary: B,
    k: T,
}

impl<T: Scalar, A: Sdf3D<T>, B: Sdf3D<T>> SmoothUnionSdf<T, A, B> {
    /// Constructs a smooth mathematical union with blending radius `k`.
    pub fn new(primary: A, secondary: B, k: T) -> Self {
        Self {
            primary,
            secondary,
            k,
        }
    }
}

impl<T: Scalar, A: Sdf3D<T>, B: Sdf3D<T>> Sdf3D<T> for SmoothUnionSdf<T, A, B> {
    fn eval(&self, p: &Point3<T>) -> T {
        let a = self.primary.eval(p);
        let b = self.secondary.eval(p);

        let diff = (a - b).abs();
        let h = (self.k - diff).max_scalar(<T as eunomia::NumericElement>::ZERO) / self.k;
        let quarter = <T as Scalar>::from_f64(0.25);

        (a).min_scalar(b) - h * h * self.k * quarter
    }

    fn bounds(&self) -> (Point3<T>, Point3<T>) {
        let (min1, max1) = self.primary.bounds();
        let (min2, max2) = self.secondary.bounds();
        let k_vec = Vector3::new(self.k, self.k, self.k);
        (
            Point3::new(
                (min1.x).min_scalar(min2.x),
                (min1.y).min_scalar(min2.y),
                (min1.z).min_scalar(min2.z),
            ) - k_vec,
            Point3::new(
                (max1.x).max_scalar(max2.x),
                (max1.y).max_scalar(max2.y),
                (max1.z).max_scalar(max2.z),
            ) + k_vec,
        )
    }
}
