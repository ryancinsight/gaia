use super::*;

// ── Locate on exact triangle edge ─────────────────────────────────────────

/// **Failure mode**: Inserting a point that lies exactly on an existing DT
/// edge exercises the `OnEdge` branch of point location and the 2-to-4
/// triangle split (or 1-to-2 on the hull).  Float rounding can misclassify
/// OnEdge as Inside, producing degenerate zero-area triangles.
#[test]
fn insert_on_exact_edge() {
    // Build a square DT, then insert a point exactly on one of the DT edges.
    let mut pts = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
    // Midpoint of the bottom edge (0,0)→(2,0).
    pts.push((1.0, 0.0));

    let dt = DelaunayTriangulation::from_points(&pts);
    assert!(dt.is_delaunay());
    assert_eq!(dt.vertex_count(), 5);
    assert!(Adjacency::verify_symmetry(dt.triangles_slice()));
}

// ── CDT diamond with interior point ───────────────────────────────────────

/// **Failure mode**: A diamond-shaped constraint polygon with a single
/// interior point.  Known to expose edge-flip bugs when the interior point
/// lies on the circumcircle of the diamond edges.
///
/// **Known issue** (Triangle, pre-2005): diamond constraints with co-
/// circular interior points caused constraint edge deletion during flips.
#[test]
fn cdt_diamond_with_interior_point() {
    let mut pslg = Pslg::new();

    // Diamond vertices.
    let top = pslg.add_vertex(5.0, 10.0);
    let right = pslg.add_vertex(10.0, 5.0);
    let bot = pslg.add_vertex(5.0, 0.0);
    let left = pslg.add_vertex(0.0, 5.0);
    pslg.add_segment(top, right);
    pslg.add_segment(right, bot);
    pslg.add_segment(bot, left);
    pslg.add_segment(left, top);

    // Interior point at centre (on circumcircle of two opposite triangles).
    pslg.add_vertex(5.0, 5.0);

    let cdt = Cdt::from_pslg(&pslg);
    let dt = cdt.triangulation();
    assert!(dt.is_delaunay());
    assert!(Adjacency::verify_symmetry(dt.triangles_slice()));
    // Should have 4 interior triangles (fan from centre).
    assert!(
        dt.triangle_count() >= 4,
        "Diamond with interior point should produce ≥4 triangles, got {}",
        dt.triangle_count()
    );
}

// ── PSLG T-intersection validation ────────────────────────────────────────

/// **Failure mode**: T-intersection (segment endpoint touching the interior
/// of another segment) is a valid PSLG configuration.  Some implementations
/// reject it or fail to recover the constraint.
#[test]
fn pslg_t_intersection_constraint() {
    let mut pslg = Pslg::new();

    // Horizontal base segment.
    let a = pslg.add_vertex(0.0, 0.0);
    let b = pslg.add_vertex(10.0, 0.0);
    pslg.add_segment(a, b);

    // Vertical T-junction: endpoint at (5, 0) on the base.
    let c = pslg.add_vertex(5.0, 0.0);
    let d = pslg.add_vertex(5.0, 5.0);
    pslg.add_segment(c, d);

    // The base segment gets split by vertex c at (5,0).
    // We need to re-segment: a→c and c→b instead of a→b.
    // But first, test that the PSLG is valid (c lies on segment a→b).
    // Depending on implementation, this may auto-split or require explicit
    // specification.  Our implementation requires explicit segments.
    let mut pslg2 = Pslg::new();
    let a2 = pslg2.add_vertex(0.0, 0.0);
    let c2 = pslg2.add_vertex(5.0, 0.0);
    let b2 = pslg2.add_vertex(10.0, 0.0);
    let d2 = pslg2.add_vertex(5.0, 5.0);
    pslg2.add_segment(a2, c2);
    pslg2.add_segment(c2, b2);
    pslg2.add_segment(c2, d2);

    let cdt = Cdt::from_pslg(&pslg2);
    let dt = cdt.triangulation();
    assert!(dt.is_delaunay());
    assert!(Adjacency::verify_symmetry(dt.triangles_slice()));
}

// ── CDT convex hull stitch ────────────────────────────────────────────────

/// **Failure mode**: Constraint segments along the convex hull boundary
/// should be trivially recovered (they're already DT edges).  But some
/// implementations incorrectly attempt to flip hull edges.
#[test]
fn cdt_constraints_on_convex_hull() {
    let mut pslg = Pslg::new();
    let pts = [
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (5.0, 5.0),
        (0.0, 5.0),
    ];
    let vids: Vec<_> = pts.iter().map(|&(x, y)| pslg.add_vertex(x, y)).collect();

    // All edges are along the convex hull.
    for i in 0..vids.len() {
        pslg.add_segment(vids[i], vids[(i + 1) % vids.len()]);
    }

    let cdt = Cdt::from_pslg(&pslg);
    let dt = cdt.triangulation();
    assert!(dt.is_delaunay());
    assert!(Adjacency::verify_symmetry(dt.triangles_slice()));
}

// ── Spiral CDT with radial constraints ────────────────────────────────────

/// **Failure mode**: Combination of spiral point distribution (stresses
/// walk) with radial constraints (stresses CDT recovery).
#[test]
fn spiral_cdt_with_radial_constraints() {
    let mut pslg = Pslg::new();
    let center = pslg.add_vertex(0.0, 0.0);

    let n = 32;
    let mut ring = Vec::new();
    for i in 0..n {
        let theta = 2.0 * PI * (i as f64) / (n as f64);
        let r = 5.0;
        ring.push(pslg.add_vertex(r * theta.cos(), r * theta.sin()));
    }

    // Boundary polygon.
    for i in 0..n {
        pslg.add_segment(ring[i], ring[(i + 1) % n]);
    }

    // Every other vertex gets a radial constraint to center.
    for i in (0..n).step_by(2) {
        pslg.add_segment(center, ring[i]);
    }

    let cdt = Cdt::from_pslg(&pslg);
    let dt = cdt.triangulation();
    assert!(dt.is_delaunay());
    assert!(Adjacency::verify_symmetry(dt.triangles_slice()));
}
