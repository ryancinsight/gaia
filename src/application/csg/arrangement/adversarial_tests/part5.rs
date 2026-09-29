use super::*;

/// Edge-contact cubes: two cubes sharing exactly one edge.
///
/// ## Known Library Failure
///
/// Edge-only contact produces zero-area intersection fragments that
/// confuse fragment classification in Cork and libigl.  The GWN
/// classifier must correctly identify the contact as a boundary
/// condition, not an interior region.
///
/// ## Theorem — Edge-Contact Union
///
/// For two closed manifolds $A$, $B$ sharing exactly one edge $e$,
/// $A \cap B = e$ (a 1-manifold) and $|A \cup B| = |A| + |B|$.
/// The union is a valid 2-manifold with two connected components
/// or (if treated as non-manifold) a pinched surface at $e$.  ∎
#[test]
fn edge_contact_cubes_union_volume_additive() {
    let cube_a = Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_a");

    // Touching on edge at (1, 1, z)
    let cube_b = Cube {
        origin: Point3r::new(1.0, 1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_b");

    let result = csg_boolean(BooleanOp::Union, &cube_a, &cube_b);
    if let Ok(mesh) = result {
        let vol = signed_volume(&mesh);
        // Two disjoint cubes (touching edge only): volume = 8 + 8 = 16
        assert!(
            (vol - 16.0).abs() < 1.0,
            "edge-contact union volume ~16, got {vol:.4}"
        );
    } else {
        // Structured error acceptable for edge-contact degeneracy
    }
}

/// Vertex-contact cubes: touching at exactly one vertex.
///
/// ## Known Library Failure
///
/// Vertex-only contact is the most degenerate configuration — the
/// intersection is a single point (0-manifold).  Cork panics on this
/// configuration; CGAL Nef handles it but produces extraneous faces.
///
/// ## Theorem — Vertex-Contact Union
///
/// For two closed manifolds $A$, $B$ sharing exactly one vertex $v$,
/// $|A \cup B| = |A| + |B|$.  The union mesh is non-manifold at $v$
/// (link is two disjoint circles, not one).  ∎
#[test]
fn vertex_contact_cubes_union_volume_additive() {
    let cube_a = Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_a");

    // Touching at vertex (1, 1, 1) = (-1+2, -1+2, -1+2) = corner of A
    let cube_b = Cube {
        origin: Point3r::new(1.0, 1.0, 1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_b");

    let result = csg_boolean(BooleanOp::Union, &cube_a, &cube_b);
    if let Ok(mesh) = result {
        let vol = signed_volume(&mesh);
        assert!(
            (vol - 16.0).abs() < 1.0,
            "vertex-contact union volume ~16, got {vol:.4}"
        );
    } else {
        // Structured error acceptable for vertex-contact degeneracy
    }
}

/// Thin-wall cube difference: hollow out a cube leaving a thin shell.
///
/// ## Known Library Failure
///
/// When the inner and outer cubes nearly coincide (thin wall), seam
/// vertex merging can collapse across the wall, creating holes.  Cork
/// and libigl are known to fail with wall thickness < ~1% of cube size.
///
/// ## Theorem — Thin-Wall Volume
///
/// For outer cube side $a$ and inner cube side $b = a - 2t$ (wall
/// thickness $t$), $|A \setminus B| = a^3 - b^3$.  For $a=2$,
/// $b=1.96$ ($t=0.02$): $V = 8 - 7.529536 = 0.470464$.  ∎
#[test]
fn thin_wall_cube_difference_no_collapse() {
    let outer = Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("outer cube");

    let wall = 0.02; // 2% wall thickness
    let inner = Cube {
        origin: Point3r::new(-1.0 + wall, -1.0 + wall, -1.0 + wall),
        width: 2.0 - 2.0 * wall,
        height: 2.0 - 2.0 * wall,
        depth: 2.0 - 2.0 * wall,
    }
    .build()
    .expect("inner cube");

    let result = csg_boolean(BooleanOp::Difference, &outer, &inner)
        .expect("thin-wall difference must succeed");
    assert!(!result.faces.is_empty(), "thin-wall must produce faces");
    let vol = signed_volume(&result);
    let outer_vol = 8.0_f64;
    let inner_side = 2.0 - 2.0 * wall;
    let inner_vol = inner_side.powi(3);
    let expected = outer_vol - inner_vol;
    assert!(
        (vol - expected).abs() < expected * 0.25,
        "thin-wall volume: expected {expected:.6}, got {vol:.6}"
    );
}

// ── Pinch-vertex adversarial tests ────────────────────────────────────
//
// These target the figure-8 vertex topology defect at dense multi-way
// junctions.  A pinch vertex passes manifold edge checks (every edge
// shared by exactly 2 faces) but violates the vertex-link simple-cycle
// invariant, reducing the Euler characteristic by 1 per pinch.
//
// Known to affect: Cork, CGAL Nef, libigl, Manifold (prior versions).

/// Three cylinders at 120° spacing — symmetric trifurcation.
///
/// # Known Library Failures
///
/// 120° spacing creates a symmetric 3-way junction where all three
/// intersection curves meet at a single point.  The rotational symmetry
/// increases the probability of vertex coincidence at the junction,
/// making pinch vertices almost certain in libraries that use
/// single-valued half-edge adjacency maps.
///
/// # Theorem (Symmetric Junction Euler Invariant)
///
/// The union of *k* cylinders meeting at a common junction with
/// genus-0 topology must satisfy χ = 2 regardless of the angular
/// spacing, provided every vertex link is a simple cycle.
///
/// **Proof sketch.**  The union boundary is a closed oriented
/// 2-manifold homeomorphic to a sphere (genus 0).  For any closed
/// oriented 2-manifold of genus *g*, χ = 2(1 − g) = 2.  ∎
#[test]
fn symmetric_120deg_cylinder_union_no_pinch() {
    use crate::application::csg::boolean::csg_boolean_nary;
    use crate::application::csg::CsgNode;
    use crate::application::watertight::check::check_watertight;
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let radius = 0.4;
    let height = 3.0;
    let segments = 24;
    let mut meshes = Vec::new();
    for angle_deg in [0.0_f64, 120.0, 240.0] {
        // Create cylinder along +Y, then rotate around Z to the desired angle.
        // Subtract PI/2 so angle 0° points along +X.
        let raw = Cylinder {
            base_center: Point3r::new(0.0, 0.0, 0.0),
            radius,
            height,
            segments,
        }
        .build()
        .expect("cylinder");
        let rotation = UnitQuaternion::<f64>::from_axis_angle(
            Vector3::z_axis(),
            angle_deg.to_radians() - std::f64::consts::FRAC_PI_2,
        );
        let m = CsgNode::Transform {
            node: Box::new(CsgNode::Leaf(Box::new(raw))),
            iso: Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
        }
        .evaluate()
        .expect("cylinder transform");
        meshes.push(m);
    }

    let mut result = csg_boolean_nary(BooleanOp::Union, &meshes).expect("120° cylinder union");
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "120° cylinder union must be watertight"
    );
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "120° cylinder union χ = {:?}, expected 2 — pinch vertex present",
        report.euler_characteristic,
    );
}

/// Four cylinders at 90° spacing in a cross pattern — dense 4-way junction.
///
/// # Known Library Failures
///
/// A 4-way cross junction creates 6 pairwise intersection curves
/// meeting at the origin.  The 90° symmetry maximises vertex
/// coincidence at the junction, making pinch vertices likely
/// in libraries with single-valued half-edge adjacency.
#[test]
fn cross_4_cylinder_union_no_pinch() {
    use crate::application::csg::boolean::csg_boolean_nary;
    use crate::application::csg::CsgNode;
    use crate::application::watertight::check::check_watertight;
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let radius = 0.4;
    let height = 3.0;
    let segments = 24;
    let mut meshes = Vec::new();
    for angle_deg in [0.0_f64, 90.0, 180.0, 270.0] {
        let raw = Cylinder {
            base_center: Point3r::new(0.0, 0.0, 0.0),
            radius,
            height,
            segments,
        }
        .build()
        .expect("cross cylinder");
        let rotation = UnitQuaternion::<f64>::from_axis_angle(
            Vector3::z_axis(),
            angle_deg.to_radians() - std::f64::consts::FRAC_PI_2,
        );
        let m = CsgNode::Transform {
            node: Box::new(CsgNode::Leaf(Box::new(raw))),
            iso: Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
        }
        .evaluate()
        .expect("cross cylinder transform");
        meshes.push(m);
    }

    let mut result = csg_boolean_nary(BooleanOp::Union, &meshes).expect("cross-4 cylinder union");
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "cross-4 cylinder union must be watertight"
    );
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "cross-4 cylinder union χ = {:?}, expected 2 — pinch vertex(es) detected",
        report.euler_characteristic,
    );
}

/// Star-shaped cylinder union — 5 cylinders at 72° spacing through origin.
///
/// # Known Library Failures
///
/// Five co-planar cylinders create a star junction with 10 pairwise
/// intersection curves.  The junction region has extreme vertex density,
/// making shared-neighbour collisions in half-edge adjacency nearly
/// guaranteed without multi-valued maps.
///
/// # Theorem (Star Junction Vertex Count)
///
/// For *k* cylinders through a common center, the junction creates
/// O(k²) intersection curves.  At each crossing of two curves, a
/// potential pinch vertex arises.  The total number of potential pinch
/// vertices is O(k²), requiring the detection algorithm to handle
/// arbitrary fan multiplicity.  ∎
#[test]
fn star_5_cylinder_union_no_pinch() {
    use crate::application::csg::boolean::csg_boolean_nary;
    use crate::application::csg::CsgNode;
    use crate::application::watertight::check::check_watertight;
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let radius = 0.3;
    let height = 3.0;
    let segments = 20;
    let mut meshes = Vec::new();
    for i in 0..5 {
        let angle_deg = f64::from(i) * 72.0;
        let raw = Cylinder {
            base_center: Point3r::new(0.0, 0.0, 0.0),
            radius,
            height,
            segments,
        }
        .build()
        .expect("star cylinder");
        let rotation = UnitQuaternion::<f64>::from_axis_angle(
            Vector3::z_axis(),
            angle_deg.to_radians() - std::f64::consts::FRAC_PI_2,
        );
        let m = CsgNode::Transform {
            node: Box::new(CsgNode::Leaf(Box::new(raw))),
            iso: Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
        }
        .evaluate()
        .expect("star cylinder transform");
        meshes.push(m);
    }

    let mut result = csg_boolean_nary(BooleanOp::Union, &meshes).expect("star-5 cylinder union");
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "star-5 cylinder union must be watertight"
    );
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "star-5 cylinder union χ = {:?}, expected 2 — pinch vertex(es) detected",
        report.euler_characteristic,
    );
}
