use super::super::basis::{eval_basis_and_deriv_to_slice, eval_basis_to_slice};
use super::super::knot::KnotVector;
use super::super::parameter::uniform_parameter;
use super::super::ratio::{rational_value, scaled_rational_term};
use super::validate::validate_surface_dims;
use super::{BSplineSurface, ControlGrid, SurfaceError, WeightGrid};
use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::{Point3, UnitVector3, Vector3};

// NurbsSurface
// ---------------------------------------------------------------------------

/// A rational tensor-product NURBS surface.
///
/// Evaluation (rational):
///   S(u,v) = (`sum_ij` `N_i(u)` * `N_j(v)` * `w_ij` * `P_ij`)
///           / (`sum_ij` `N_i(u)` * `N_j(v)` * `w_ij`)
///
/// Exact conics (spheres, cylinders) arise from specific weight configurations.
#[derive(Clone, Debug)]
pub struct NurbsSurface<T = Real> {
    /// Control point grid.
    pub control_grid: ControlGrid<T>,
    /// Positive weight at each control point.
    pub weights: WeightGrid<T>,
    /// Knot vector in the u direction.
    pub knots_u: KnotVector<T>,
    /// Knot vector in the v direction.
    pub knots_v: KnotVector<T>,
    /// Degree in the u direction.
    pub degree_u: usize,
    /// Degree in the v direction.
    pub degree_v: usize,
}

impl<T: Scalar> NurbsSurface<T> {
    /// Create a NURBS surface, validating all dimensions.
    ///
    pub fn new(
        control_grid: ControlGrid<T>,
        weights: WeightGrid<T>,
        knots_u: KnotVector<T>,
        knots_v: KnotVector<T>,
        degree_u: usize,
        degree_v: usize,
    ) -> Result<Self, SurfaceError> {
        validate_surface_dims(&control_grid, &knots_u, &knots_v, degree_u, degree_v)?;
        if weights.n_rows() != control_grid.n_rows() || weights.n_cols() != control_grid.n_cols() {
            return Err(SurfaceError::WeightGridMismatch);
        }
        Ok(Self {
            control_grid,
            weights,
            knots_u,
            knots_v,
            degree_u,
            degree_v,
        })
    }

    /// Create a NURBS surface from a B-spline (all weights = 1).
    #[must_use]
    pub fn from_bspline(s: BSplineSurface<T>) -> Self {
        let w = WeightGrid::uniform(s.control_grid.n_rows(), s.control_grid.n_cols());
        Self {
            control_grid: s.control_grid,
            weights: w,
            knots_u: s.knots_u,
            knots_v: s.knots_v,
            degree_u: s.degree_u,
            degree_v: s.degree_v,
        }
    }

    /// Create with automatic clamped uniform knot vectors and uniform weights.
    pub fn clamped(
        control_grid: ControlGrid<T>,
        degree_u: usize,
        degree_v: usize,
    ) -> Result<Self, SurfaceError> {
        let s = BSplineSurface::clamped(control_grid, degree_u, degree_v)?;
        Ok(Self::from_bspline(s))
    }

    /// Parameter domain `((u_min, u_max), (v_min, v_max))`.
    #[must_use]
    pub fn domain(&self) -> ((T, T), (T, T)) {
        (self.knots_u.domain(), self.knots_v.domain())
    }

    /// Evaluate the NURBS surface at `(u, v)`.
    #[must_use]
    pub fn point(&self, u: T, v: T) -> Point3<T> {
        self.rational_point(u, v)
    }

    /// Evaluate surface point and partial derivatives `(S, dS/du, dS/dv)`.
    ///
    /// The quotient rule is evaluated in difference form:
    /// `dS/du = sum(dN_i/du * N_j * w_ij * (P_ij - S)) / W`, with the
    /// corresponding `N_i * dN_j/dv` coefficient for `dS/dv`; `W` is the
    /// rational basis-weight sum. Finite factors and coordinate differences
    /// are scaled before multiplication or subtraction to preserve
    /// representable partial components.
    #[must_use]
    pub fn point_and_derivs(&self, u: T, v: T) -> (Point3<T>, Vector3<T>, Vector3<T>) {
        let n_u = self.control_grid.n_cols() - 1; // u = cols
        let n_v = self.control_grid.n_rows() - 1; // v = rows
        let su = self.knots_u.find_span(u, n_u);
        let sv = self.knots_v.find_span(v, n_v);
        let zero = <T as NumericElement>::ZERO;
        let mut bu_buf = [zero; 9];
        let mut dbu_buf = [zero; 9];
        let mut bv_buf = [zero; 9];
        let mut dbv_buf = [zero; 9];
        let mut bu_vec;
        let mut dbu_vec;
        let mut bv_vec;
        let mut dbv_vec;
        let (bu, dbu) = if self.degree_u <= 8 {
            (
                &mut bu_buf[..=self.degree_u],
                &mut dbu_buf[..=self.degree_u],
            )
        } else {
            bu_vec = vec![zero; self.degree_u + 1];
            dbu_vec = vec![zero; self.degree_u + 1];
            (&mut bu_vec[..], &mut dbu_vec[..])
        };
        let (bv, dbv) = if self.degree_v <= 8 {
            (
                &mut bv_buf[..=self.degree_v],
                &mut dbv_buf[..=self.degree_v],
            )
        } else {
            bv_vec = vec![zero; self.degree_v + 1];
            dbv_vec = vec![zero; self.degree_v + 1];
            (&mut bv_vec[..], &mut dbv_vec[..])
        };
        eval_basis_and_deriv_to_slice(su, u, self.degree_u, &self.knots_u, bu, dbu);
        eval_basis_and_deriv_to_slice(sv, v, self.degree_v, &self.knots_v, bv, dbv);

        let (s, w, exponent) = self.rational_value(su, sv, bu, bv);
        if w == zero {
            return (
                self.control_grid.get(0, 0),
                Vector3::<T>::zeros(),
                Vector3::<T>::zeros(),
            );
        }
        let (u_start, v_start) = (su - self.degree_u, sv - self.degree_v);
        let mut ds_du = Vector3::<T>::zeros();
        let mut ds_dv = Vector3::<T>::zeros();
        for (j, (&nu, &dnu)) in bu.iter().zip(dbu.iter()).enumerate() {
            for (k, (&nv, &dnv)) in bv.iter().zip(dbv.iter()).enumerate() {
                if (dnu.abs() <= zero || nv.abs() <= zero)
                    && (nu.abs() <= zero || dnv.abs() <= zero)
                {
                    continue;
                }
                let point = self.control_grid.get(v_start + k, u_start + j);
                let surface_point = Point3::from(s);
                if point == surface_point {
                    continue;
                }
                let wij = self.weights.get(v_start + k, u_start + j);
                if dnu.abs() > zero && nv.abs() > zero {
                    add_rational_derivative(
                        &mut ds_du,
                        point,
                        surface_point,
                        [dnu, nv],
                        wij,
                        [<T as NumericElement>::ONE, w],
                        -exponent,
                    );
                }
                if nu.abs() > zero && dnv.abs() > zero {
                    add_rational_derivative(
                        &mut ds_dv,
                        point,
                        surface_point,
                        [nu, dnv],
                        wij,
                        [<T as NumericElement>::ONE, w],
                        -exponent,
                    );
                }
            }
        }
        (Point3::from(s), ds_du, ds_dv)
    }

    /// Unit surface normal at `(u, v)`.
    /// Returns `None` if degenerate (zero cross product).
    #[must_use]
    pub fn normal(&self, u: T, v: T) -> Option<UnitVector3<T>> {
        let (_, du, dv) = self.point_and_derivs(u, v);
        UnitVector3::try_new(du.cross(dv), <T as Scalar>::from_f64(1e-15))
    }

    /// Axis-aligned bounding box from a resolution x resolution sample grid.
    #[must_use]
    pub fn aabb(&self, resolution: usize) -> crate::domain::geometry::Aabb<T> {
        use crate::domain::geometry::Aabb;
        let mut aabb = Aabb::<T>::empty();
        let ((u0, u1), (v0, v1)) = self.domain();
        let res = resolution.max(4);
        for i in 0..=res {
            let u = uniform_parameter(u0, u1, i, res);
            for j in 0..=res {
                let v = uniform_parameter(v0, v1, j, res);
                let pt = self.point(u, v);
                aabb.expand(&pt);
            }
        }
        aabb
    }

    // -- internal --

    fn rational_point(&self, u: T, v: T) -> Point3<T> {
        let n_u = self.control_grid.n_cols() - 1; // u = cols
        let n_v = self.control_grid.n_rows() - 1; // v = rows
        let su = self.knots_u.find_span(u, n_u);
        let sv = self.knots_v.find_span(v, n_v);
        let zero = <T as NumericElement>::ZERO;
        let mut bu_buf = [zero; 9];
        let mut bv_buf = [zero; 9];
        let mut bu_vec;
        let mut bv_vec;
        let bu = if self.degree_u <= 8 {
            &mut bu_buf[..=self.degree_u]
        } else {
            bu_vec = vec![zero; self.degree_u + 1];
            &mut bu_vec[..]
        };
        let bv = if self.degree_v <= 8 {
            &mut bv_buf[..=self.degree_v]
        } else {
            bv_vec = vec![zero; self.degree_v + 1];
            &mut bv_vec[..]
        };
        eval_basis_to_slice(su, u, self.degree_u, &self.knots_u, bu);
        eval_basis_to_slice(sv, v, self.degree_v, &self.knots_v, bv);

        Point3::from(self.rational_value(su, sv, bu, bv).0)
    }

    fn rational_value(&self, su: usize, sv: usize, bu: &[T], bv: &[T]) -> (Vector3<T>, T, i32) {
        let zero = <T as NumericElement>::ZERO;
        let (u_start, v_start) = (su - self.degree_u, sv - self.degree_v);
        let terms = bv.iter().enumerate().flat_map(|(k, &nv)| {
            bu.iter().enumerate().filter_map(move |(j, &nu)| {
                (nu.abs() > zero && nv.abs() > zero).then_some((
                    [nu, nv, self.weights.get(v_start + k, u_start + j)],
                    self.control_grid.get(v_start + k, u_start + j).coords.data,
                ))
            })
        });
        let (point, denominator, exponent) =
            rational_value(terms, self.control_grid.get(0, 0).coords.data);
        (Vector3::from(point), denominator, exponent)
    }
}

fn add_rational_derivative<T: Scalar>(
    derivative: &mut Vector3<T>,
    point: Point3<T>,
    surface_point: Point3<T>,
    basis: [T; 2],
    raw_weight: T,
    denominator: [T; 2],
    exponent_offset: i32,
) {
    let factors = [basis[0], basis[1], raw_weight];
    for (component, (positive, negative)) in derivative
        .data
        .iter_mut()
        .zip(point.coords.data.into_iter().zip(surface_point.coords.data))
    {
        if positive != negative {
            *component +=
                scaled_rational_term(factors, denominator, positive, negative, exponent_offset);
        }
    }
}

// ---------------------------------------------------------------------------
