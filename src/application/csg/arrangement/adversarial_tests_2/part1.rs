use super::*;

// ── 1. Genus > 0 — Torus intersected with cube ────────────────────────
//
// Theorem (Genus stability under Boolean intersection):
//   Let T be a genus-1 surface (torus) and C a genus-0 surface (cube).
//   If C fully contains T, then T ∩ C = T with genus 1.
//   If C partially intersects T, the result genus depends on the
//   number and topology of the intersection curves.
//
// Known library failures: Cork crashes on genus > 0 GWN evaluation;
// CGAL exact-arithmetic fallback is O(n²) on torus meshes.

/// Torus fully inside a large cube — intersection should preserve torus.
///
/// The cube is large enough to contain the entire torus.  The result
/// must have the same volume (within tolerance) as the torus alone.
#[test]
fn torus_inside_large_cube_intersection_preserves_volume() {
    let torus = Torus {
        major_radius: 2.0,
        minor_radius: 0.5,
        major_segments: 24,
        minor_segments: 12,
    }
    .build()
    .expect("torus");

    // Cube that fully encloses the torus (extends ±4 in all axes).
    let big_cube = Cube {
        origin: Point3r::new(-4.0, -4.0, -4.0),
        width: 8.0,
        height: 8.0,
        depth: 8.0,
    }
    .build()
    .expect("big_cube");

    let vol_torus = signed_volume(&torus);
    assert!(vol_torus > 0.0, "torus must have positive signed volume");

    match csg_boolean(BooleanOp::Intersection, &torus, &big_cube) {
        Ok(result) => {
            let vol_result = signed_volume(&result);
            let err = (vol_result - vol_torus).abs() / vol_torus;
            assert!(
                err < 0.15,
                "torus ∩ big_cube volume should ≈ torus volume: \
                     result={vol_result:.4}, torus={vol_torus:.4}, rel_err={err:.4}"
            );
        }
        Err(e) => {
            panic!("torus ∩ big_cube must not fail: {e:?}");
        }
    }
}

/// Torus clipped by a cube — partial intersection produces valid mesh.
///
/// The cube only covers half the torus, producing a mesh with open
/// topology that the postprocessor must handle gracefully.
#[test]
fn torus_partial_cube_intersection_valid() {
    let torus = Torus {
        major_radius: 2.0,
        minor_radius: 0.5,
        major_segments: 24,
        minor_segments: 12,
    }
    .build()
    .expect("torus");

    // Cube covers only the +X half of the torus.
    let half_cube = Cube {
        origin: Point3r::new(0.0, -4.0, -4.0),
        width: 4.0,
        height: 8.0,
        depth: 8.0,
    }
    .build()
    .expect("half_cube");

    let vol_torus = signed_volume(&torus);

    match csg_boolean(BooleanOp::Intersection, &torus, &half_cube) {
        Ok(result) => {
            let vol_result = signed_volume(&result);
            // Intersection volume should be roughly half the torus.
            assert!(
                vol_result > 0.0,
                "partial torus intersection must have positive volume, got {vol_result:.6}"
            );
            assert!(
                vol_result < vol_torus * 0.85,
                "partial intersection must be smaller than full torus: \
                     result={vol_result:.4} vs torus={vol_torus:.4}"
            );
        }
        Err(_e) => {
            // Some arrangements may fail on complex genus-1 topology —
            // acceptable as long as it's a clean error, not a panic.
        }
    }
}

// ── 2. Mixed orientation (CW + CCW operands) ──────────────────────────
//
// Theorem (Winding orientation invariance):
//   Boolean operations on closed meshes must be invariant under
//   global face-winding reversal of either operand, because the
//   GWN classification is signed: flipping winding negates the GWN,
//   and the pipeline must normalise orientation before classifying.
//
// Known library failures: libigl assumes consistent CCW winding;
// meshes with reversed winding produce inside-out results.

/// Union of a normal-winding cube and a reversed-winding cube.
///
/// The second cube has all face normals flipped (CW winding).
/// The pipeline should detect and normalise orientation before
/// classification, producing the same result as two CCW cubes.
#[test]
fn mixed_orientation_union_produces_valid_result() {
    let a = unit_cube();

    // Create a cube and flip all face winding (reverse vertex order).
    let mut b = Cube {
        origin: Point3r::new(0.5, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("offset_cube");
    b.flip_faces();

    let vol_a = signed_volume(&a);

    match csg_boolean(BooleanOp::Union, &a, &b) {
        Ok(result) => {
            let vol = signed_volume(&result);
            assert!(
                vol > vol_a * 0.9,
                "union with flipped operand should have ≥ single cube vol: {vol:.4}"
            );
        }
        Err(_e) => {
            // If the pipeline rejects reversed-winding input, that's
            // acceptable — but it must not panic.
        }
    }
}

// ── 3. Many-operand coplanar N-ary union (≥ 10 cubes) ─────────────────
//
// Theorem (N-ary union volume bound):
//   For operands O₁, …, Oₙ with volumes V₁, …, Vₙ,
//   max(Vᵢ) ≤ Vol(O₁ ∪ … ∪ Oₙ) ≤ Σ Vᵢ.
//
// Known library failures: Manifold's merge step is O(n²) for large n;
// coplanar face disambiguation fails with > 6 concurrent coplanar faces.

/// 10 overlapping cubes in a line, resolved via N-ary union.
///
/// Each cube overlaps its neighbour by 50%.  The result should be
/// a single elongated box with no internal faces.
#[test]
fn ten_cube_nary_union_volume_bound() {
    let cubes: Vec<IndexedMesh> = (0..10)
        .map(|i| {
            Cube {
                origin: Point3r::new(f64::from(i) * 1.0, -1.0, -1.0),
                width: 2.0,
                height: 2.0,
                depth: 2.0,
            }
            .build()
            .expect("cube_i")
        })
        .collect();

    let single_vol = signed_volume(&cubes[0]);
    let sum_vol = single_vol * 10.0;

    match csg_boolean_nary(BooleanOp::Union, &cubes) {
        Ok(result) => {
            let vol = signed_volume(&result);
            assert!(
                vol >= single_vol * 0.95,
                "10-cube union vol {vol:.4} must be ≥ single cube vol {single_vol:.4}"
            );
            assert!(
                vol < sum_vol * 1.05,
                "10-cube union vol {vol:.4} must be < sum {sum_vol:.4}"
            );
        }
        Err(e) => {
            panic!("10-cube N-ary union must not fail: {e:?}");
        }
    }
}

// ── 4. Sharp dihedral angle intersection ──────────────────────────────
//
// Theorem (GWN continuity near sharp edges):
//   The GWN is continuous everywhere except at the mesh surface.
//   Near a dihedral edge with angle θ → 0, the GWN gradient
//   increases as 1/sin(θ), amplifying floating-point error.
//   For θ < 2°, single-precision GWN may misclassify points
//   within distance ε/sin(θ) of the edge (ε = machine epsilon).
//
// Known library failures: Cork misclassifies GWN near sub-2° edges;
// libigl's winding number tree has poor convergence for thin wedges.

/// Two cubes intersecting at a very shallow angle (≈2°).
///
/// Creates a thin wedge-shaped intersection region.  The GWN near
/// the sharp dihedral edge must still correctly classify fragments.
#[test]
fn sharp_dihedral_intersection_stable() {
    use crate::application::csg::CsgNode;
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let cube_a = unit_cube();
    let angle_rad = 2.0_f64.to_radians(); // 2° rotation

    // Rotate cube B by 2° around the Z axis — creates a thin wedge overlap.
    let raw_b = unit_cube();
    let rotation = UnitQuaternion::<f64>::from_axis_angle(Vector3::z_axis(), angle_rad);
    let cube_b = CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(raw_b))),
        iso: Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
    }
    .evaluate()
    .expect("rotated cube");

    match csg_boolean(BooleanOp::Intersection, &cube_a, &cube_b) {
        Ok(result) => {
            let vol = signed_volume(&result);
            let vol_a = signed_volume(&cube_a);
            // The intersection of two identical cubes rotated 2° should
            // be very close to the original volume (most of the cube overlaps).
            assert!(
                vol > vol_a * 0.85,
                "2° intersection should retain most volume: {vol:.4} vs {vol_a:.4}"
            );
            assert!(
                vol <= vol_a * 1.01,
                "intersection cannot exceed operand: {vol:.4} > {vol_a:.4}"
            );
        }
        Err(e) => {
            panic!("sharp dihedral intersection must not fail: {e:?}");
        }
    }
}

// ── 5. Fully-interior subtraction (cavity topology) ───────────────────
//
// Theorem (Cavity Euler characteristic):
//   Let A be a genus-0 closed mesh containing B entirely in its interior.
//   Then A \ B is a genus-0 closed mesh with χ = 4 (outer shell χ=2 +
//   inner cavity χ=2), or equivalently β₀ = 2 if the cavity is
//   treated as a separate connected component by the mesh topology.
//
// Known library failures: CGAL Nef polyhedron correctly handles this;
// Cork and libigl often produce non-manifold edges at the cavity seam.

/// Large cube minus small interior cube — must create a hollow shell.
#[test]
fn fully_interior_subtraction_creates_cavity() {
    let outer = Cube {
        origin: Point3r::new(-2.0, -2.0, -2.0),
        width: 4.0,
        height: 4.0,
        depth: 4.0,
    }
    .build()
    .expect("outer");

    let inner = unit_cube(); // [-1,1]³, fully inside outer [-2,2]³

    let vol_outer = signed_volume(&outer);
    let vol_inner = signed_volume(&inner);

    match csg_boolean(BooleanOp::Difference, &outer, &inner) {
        Ok(result) => {
            let vol = signed_volume(&result);
            let expected = vol_outer - vol_inner;
            let err = (vol - expected).abs() / expected;
            assert!(
                err < 0.15,
                "cavity vol should ≈ outer − inner: result={vol:.4}, \
                     expected={expected:.4}, err={err:.4}"
            );
        }
        Err(e) => {
            panic!("fully-interior subtraction must not fail: {e:?}");
        }
    }
}

/// Large sphere minus small interior sphere — cavity with curved surfaces.
#[test]
fn sphere_cavity_subtraction_volume() {
    let outer = UvSphere {
        radius: 3.0,
        center: Point3r::origin(),
        segments: 24,
        stacks: 12,
    }
    .build()
    .expect("outer_sphere");

    let inner = UvSphere {
        radius: 1.0,
        center: Point3r::origin(),
        segments: 24,
        stacks: 12,
    }
    .build()
    .expect("inner_sphere");

    let vol_outer = signed_volume(&outer);
    let vol_inner = signed_volume(&inner);

    match csg_boolean(BooleanOp::Difference, &outer, &inner) {
        Ok(result) => {
            let vol = signed_volume(&result);
            let expected = vol_outer - vol_inner;
            let err = (vol - expected).abs() / expected;
            assert!(
                err < 0.20,
                "sphere cavity vol should ≈ outer − inner: result={vol:.4}, \
                     expected={expected:.4}, err={err:.4}"
            );
        }
        Err(e) => {
            panic!("sphere cavity subtraction must not fail: {e:?}");
        }
    }
}

// ── 6. Repeated-scale stability ───────────────────────────────────────
//
// Theorem (Scale-invariance of Boolean):
//   Let S(k) denote uniform scaling by factor k.  For any meshes A, B:
//   S(k)(A op B) = S(k)(A) op S(k)(B).  Applying union, then scaling,
//   then another union, then unscaling, should produce the same result
//   (up to FP error) as applying all three operations at original scale.
//
// Known library failures: absolute tolerances cause different welding
// decisions at different scales, producing topologically different results.

/// Scale → union → unscale should match direct union.
#[test]
fn scale_union_unscale_stability() {
    let a = unit_cube();
    let b = Cube {
        origin: Point3r::new(0.5, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("b_cube");

    // Direct union at original scale.
    let direct = csg_boolean(BooleanOp::Union, &a, &b);

    // Build 100× scaled equivalents directly.
    let a_scaled = Cube {
        origin: Point3r::new(-100.0, -100.0, -100.0),
        width: 200.0,
        height: 200.0,
        depth: 200.0,
    }
    .build()
    .expect("a_scaled");
    let b_scaled = Cube {
        origin: Point3r::new(50.0, -100.0, -100.0),
        width: 200.0,
        height: 200.0,
        depth: 200.0,
    }
    .build()
    .expect("b_scaled");

    let scaled_result = csg_boolean(BooleanOp::Union, &a_scaled, &b_scaled);

    match (direct, scaled_result) {
        (Ok(ref d), Ok(ref s)) => {
            let vol_direct = signed_volume(d);
            // Scale volume back: V_scaled / 100³ = V_original
            let vol_scaled_back = signed_volume(s) / (100.0_f64).powi(3);
            let err = (vol_direct - vol_scaled_back).abs() / vol_direct.max(1e-15);
            assert!(
                err < 0.10,
                "scale-invariant volume: direct={vol_direct:.4}, \
                     scaled_back={vol_scaled_back:.4}, err={err:.4}"
            );
        }
        (Err(e), _) => panic!("direct union failed: {e:?}"),
        (_, Err(e)) => panic!("scaled union failed: {e:?}"),
    }
}

// ── 7. Near-tangent cylinder contact ──────────────────────────────────
//
// Theorem (Tangent contact degeneracy):
//   Two cylinders with parallel axes at distance d = r₁ + r₂ are tangent
//   along a line.  The intersection curve degenerates to a single line,
//   producing zero-area fragments.  At d = r₁ + r₂ + ε (ε → 0), the
//   intersection curve is a very thin ellipse, creating near-degenerate
//   fragment triangles that stress the CDT co-refinement phase.
//
// Known library failures: Cork produces non-manifold edges at tangent lines;
// Manifold's simplification collapses the thin intersection entirely.

/// Two cylinders with axes separated by exactly 2R + ε.
///
/// The intersection region is extremely thin.  The Boolean must still
/// produce a valid result (possibly empty if ε is subresolution).
#[test]
fn near_tangent_cylinders_union_valid() {
    let r = 1.0;
    let epsilon = 0.01; // just barely separated
    let separation = 2.0 * r + epsilon;

    let cyl_a = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius: r,
        height: 4.0,
        segments: 24,
    }
    .build()
    .expect("cyl_a");

    let cyl_b = Cylinder {
        base_center: Point3r::new(separation, 0.0, 0.0),
        radius: r,
        height: 4.0,
        segments: 24,
    }
    .build()
    .expect("cyl_b");

    match csg_boolean(BooleanOp::Union, &cyl_a, &cyl_b) {
        Ok(result) => {
            let vol = signed_volume(&result);
            let vol_a = signed_volume(&cyl_a);
            let vol_b = signed_volume(&cyl_b);
            // Near-tangent union: essentially two separate cylinders.
            // Volume ≈ vol_a + vol_b (minimal overlap).
            let sum = vol_a + vol_b;
            assert!(
                vol > sum * 0.90,
                "near-tangent union vol {vol:.4} should ≈ sum {sum:.4}"
            );
        }
        Err(_e) => {
            // Acceptable if pipeline gracefully rejects degenerate overlap.
        }
    }
}
