use super::*;

// ── 8. Self-intersection detection on crafted non-manifold input ──────
//
// Theorem (Non-manifold detectability):
//   The detect_self_intersections function implements Möller (1997)
//   triangle-triangle intersection.  It detects face pairs whose
//   interiors cross in 3D (non-coplanar, non-adjacent).  Coplanar
//   overlaps return false by design (step 3 of Möller's algorithm).
//
// Known library failures: most libraries simply crash or hang on
// non-manifold input rather than detecting and reporting it.

/// Two non-adjacent crossing triangles in 3D — must detect intersection.
#[test]
fn crossing_triangles_detected() {
    let mut pool = VertexPool::default_millifluidic();
    let n = leto::geometry::Vector3::zeros();
    // Triangle A in z=0 plane, centred at origin.
    let a0 = pool.insert_or_weld(Point3r::new(-2.0, -2.0, 0.0), n);
    let a1 = pool.insert_or_weld(Point3r::new(2.0, -2.0, 0.0), n);
    let a2 = pool.insert_or_weld(Point3r::new(0.0, 2.0, 0.0), n);
    // Triangle B in y=0 plane, crossing A.
    let b0 = pool.insert_or_weld(Point3r::new(-2.0, 0.0, -2.0), n);
    let b1 = pool.insert_or_weld(Point3r::new(2.0, 0.0, -2.0), n);
    let b2 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 2.0), n);

    let faces = vec![
        FaceData::untagged(a0, a1, a2),
        FaceData::untagged(b0, b1, b2),
    ];

    let pairs = detect_self_intersections(&faces, &pool);
    assert!(
        !pairs.is_empty(),
        "crossing triangles in 3D must be detected as self-intersecting"
    );
}

/// Four triangles forming a "bowtie" — two pairs cross in 3D.
#[test]
fn bowtie_crossing_detected() {
    let mut pool = VertexPool::default_millifluidic();
    let n = leto::geometry::Vector3::zeros();
    // Two triangles in the XZ plane.
    let a0 = pool.insert_or_weld(Point3r::new(-1.0, 0.0, -1.0), n);
    let a1 = pool.insert_or_weld(Point3r::new(1.0, 0.0, -1.0), n);
    let a2 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 1.0), n);
    // Triangle B crosses A through the XY plane.
    let b0 = pool.insert_or_weld(Point3r::new(-1.0, -1.0, 0.0), n);
    let b1 = pool.insert_or_weld(Point3r::new(1.0, -1.0, 0.0), n);
    let b2 = pool.insert_or_weld(Point3r::new(0.0, 1.0, 0.0), n);

    let faces = vec![
        FaceData::untagged(a0, a1, a2),
        FaceData::untagged(b0, b1, b2),
    ];

    let pairs = detect_self_intersections(&faces, &pool);
    assert!(!pairs.is_empty(), "bowtie crossing must be detected");
}

// ── 9. N-ary intersection of multiple cubes ───────────────────────────
//
// Theorem (N-ary intersection volume monotonicity):
//   Vol(A₁ ∩ … ∩ Aₙ) ≤ Vol(A₁ ∩ … ∩ Aₙ₋₁) for any additional Aₙ.
//   I.e. intersecting with one more operand can only reduce volume.
//
// This tests the N-ary path for intersection (not just union),
// which exercises a different code path in fragment survivorship.

/// 5 cubes with progressive offsets — intersection shrinks as expected.
#[test]
fn five_cube_nary_intersection_shrinks() {
    let cubes: Vec<IndexedMesh> = (0..5)
        .map(|i| {
            let offset = f64::from(i) * 0.3;
            Cube {
                origin: Point3r::new(-1.0 + offset, -1.0, -1.0),
                width: 2.0,
                height: 2.0,
                depth: 2.0,
            }
            .build()
            .expect("cube_i")
        })
        .collect();

    let vol_single = signed_volume(&cubes[0]);

    match csg_boolean_nary(BooleanOp::Intersection, &cubes) {
        Ok(result) => {
            let vol = signed_volume(&result);
            assert!(
                vol > 0.0,
                "5-cube intersection should be non-empty: {vol:.6}"
            );
            assert!(
                vol < vol_single * 0.95,
                "5-cube intersection must be smaller than single cube: \
                     {vol:.4} vs {vol_single:.4}"
            );
        }
        Err(e) => {
            panic!("5-cube N-ary intersection must not fail: {e:?}");
        }
    }
}

// ── 10. Cube-sphere intersection — curved + flat face interaction ─────

/// Cube clipping a sphere — the intersection curve is a circle
/// embedded in the cube face.  Tests curved-flat co-refinement.
#[test]
fn cube_sphere_clip_intersection_valid() {
    let cube = unit_cube();
    let sphere = UvSphere {
        radius: 1.5,
        center: Point3r::origin(),
        segments: 24,
        stacks: 12,
    }
    .build()
    .expect("sphere");

    let vol_cube = signed_volume(&cube);

    match csg_boolean(BooleanOp::Intersection, &cube, &sphere) {
        Ok(result) => {
            let vol = signed_volume(&result);
            // Intersection is the part of the cube inside the sphere.
            // Since the sphere radius (1.5) > cube half-width (1.0),
            // most of the cube is inside the sphere.
            assert!(
                vol > vol_cube * 0.5,
                "cube-sphere intersection should retain significant volume: {vol:.4}"
            );
            assert!(
                vol <= vol_cube * 1.01,
                "intersection cannot exceed cube volume: {vol:.4}"
            );
        }
        Err(e) => {
            panic!("cube-sphere intersection must not fail: {e:?}");
        }
    }
}
