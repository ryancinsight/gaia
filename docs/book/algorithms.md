# Algorithm Reference

This chapter documents the key algorithms Gaia uses with their theoretical
guarantees, practical bounds, and relationships to the generated gallery figures.

---

## Exact Geometric Predicates

Gaia uses adaptive-precision arithmetic for all orientation and in-circle
decisions. The two foundations are:

- **`orient_2d(a,b,c)`** — returns the sign of the signed area of triangle
  `(a,b,c)`. Positive means CCW, Negative means CW, Degenerate means exactly
  collinear.
- **`orient_3d(a,b,c,d)`** — returns the sign of the scalar triple product
  `(b−a)·((c−a)×(d−a))`. Used for tetrahedral volume sign and half-space tests.

### Why exact predicates matter for CSG

A Boolean union of two slightly touching cubes produces a *shared edge* whose
orientation can flip between `Positive` and `Negative` depending on floating-
point rounding. Without exact predicates, the CSG kernel would misclassify the
edge as a self-intersection and either silently omit the face or produce a
non-manifold result. Exact arithmetic ensures the classification is
deterministic regardless of coordinate magnitude or compiler optimizations.

### Where exactness ends

Exact predicates decide sign. They do not decide CSG inside/outside membership.
`classify_fragment` uses the Generalized Winding Number with symmetric
thresholds `GWN_OUTSIDE_THRESHOLD = 0.25` and `GWN_INSIDE_THRESHOLD = 0.75`
(see ADR-0004), then resolves the `[0.25, 0.75]` band with coplanarity and
nearest-face tiebreakers (see ADR-0005).

---

## Constrained Delaunay Triangulation (CDT)

CDT is used in three places:

1. **2-D polygon Boolean operations** (`clip/polygon2d/cdt.rs`) — the canonical
   production backend for coplanar face overlap.
2. **CDT co-refinement** (`corefine.rs`) — splitting intersecting face pairs into
   a shared planar arrangement.
3. **Ruppert's algorithm** (`delaunay/dim2/refinement/`) — quality-controlled
   surface triangulation.

### Bowyer–Watson insertion

The kernel inserts one point at a time by:
1. Finding all triangles whose circumcircle contains the new point (the *conflict
   region*).
2. Removing the conflict triangles to expose a star-shaped cavity.
3. Connecting the new point to every edge of the cavity.

**Correctness**: the empty-circumsphere property is restored after each insertion
because the Delaunay condition holds for all edges of the cavity before the new
point is added, and the reconnection only creates edges that satisfy it
(Shewchuk 1997, Lemma 2.1). ∎

### Ruppert's refinement termination

Ruppert's algorithm inserts circumcenters for any triangle whose radius-edge
ratio `ρ = R / l_min > B` (the bound). It terminates because:

1. Every circumcenter insertion either *splits an encroached segment* (making
   the PSLG smaller) or *strictly increases the minimum angle*.
2. The minimum angle is bounded above by the geometry; therefore the algorithm
   terminates in finite steps.

**Bound**: for B ≥ √2 ≈ 1.414, the algorithm is guaranteed to terminate and
produces a mesh with all angles ≥ arcsin(1/(2B)) ≥ 20.7°
(Ruppert 1995, Theorem 4.1).

---

## Generalized Winding Number (GWN)

The GWN classifies a 3-D query point against a triangle soup without requiring
the soup to be manifold. For each query point `q`:

```text
w(q) = (1/4π) Σ_i Ω_i(q)
```

where `Ω_i` is the solid angle subtended by triangle `i` at `q`.

**Properties:**
- For a closed, outward-oriented surface: `w(q) = 1` inside, `0` outside.
- For an open surface or inconsistently wound mesh: `w(q) ∈ ℝ`.
- Scale invariant: `w(λq) = w(q)` for any scaling `λ`.
- The van Oosterom–Strackee formula avoids expensive arcsine computation:
  `Ω = 2 atan2(det([a,b,c]), 1 + a·b + b·c + c·a)`.

Gaia's implementation in the gallery figures:
- Triangle soups with consistent winding produce clean GWN classification
  as shown in the cube and sphere panels.
- The watertightness diagnostics show how boundary edges and non-manifold
  edges corrupt the winding number: see [Watertightness diagnostics](watertightness.md).

---

## Hexahedron → Tetrahedron Decomposition

The `HexToTetConverter` decomposes each hexahedron into either 5 or 6 tetrahedra
using quality-maximizing patterns.

### Five-tet pattern

```text
[v0,v1,v3,v4], [v1,v2,v3,v6], [v4,v7,v6,v3],
[v4,v6,v5,v1], [v1,v3,v4,v6]
```

### Six-tet pattern (fan)

Fan from vertex v0 through the opposite face:
```text
[v0,v1,v2,v6], [v0,v2,v3,v6], [v0,v3,v7,v6],
[v0,v7,v4,v6], [v0,v4,v5,v6], [v0,v5,v1,v6]
```

The converter selects the pattern with the larger minimum tet volume (better
quality). **Winding correction** (Phase 34, 2026-10-05): for a positive-volume
tet `[v0,v1,v2,v3]`, the outward face normals are:

| Face (opposite) | Winding |
|---|---|
| opposite v3 | `[v0, v2, v1]` |
| opposite v2 | `[v0, v1, v3]` |
| opposite v1 | `[v0, v3, v2]` |
| opposite v0 | `[v1, v2, v3]` |

The gallery panel confirms correct boundary rendering after this fix.

![Hex to tet](figures/models/topology/hex-to-tet.svg)

---

## Watertightness Verification

A surface mesh is **watertight** if:

1. **Manifold + closed**: every edge is shared by exactly 2 faces.
2. **Euler characteristic** `V − E + F = 2(1 − g)` where `g` is the genus.
   A sphere or cube has `g = 0` → characteristic = 2.
3. **Orientation consistency**: no two adjacent faces have the same directed-edge
   orientation.
4. **Positive signed volume**: the divergence-theorem integral
   `V = (1/6) |Σ_i (a_i × b_i)·c_i|` is finite and positive.

For a triangle mesh: `E = 3F/2` (each interior edge shared by 2 faces), so the
Euler relation becomes `V − F/2 = 2`.

See the [watertightness diagnostics](watertightness.md) for visual examples of
each failure mode, and the [Watertightness figure manifest](watertightness_manifest.md)
for exact diagnostic values.

---

## Surface Normal Orientation Repair (`orient_outward`)

After mesh construction or Boolean operations, some surface faces may have
inward-pointing normals. `orient_outward` repairs this using BFS flood:

1. Identify the face with the largest X-coordinate centroid (the *extremal face*).
   For a convex solid, this face unambiguously points outward.
2. BFS: propagate orientation from the seed. Two faces sharing directed edge
   `(A→B)` and `(B→A)` have consistent winding; `(A→B)` and `(A→B)` (same
   direction) indicates a flip.
3. Check the signed-volume integral. If negative, all labels are flipped globally
   (the seed heuristic was wrong for a fully inward-wound mesh).

**Complexity**: O(F + E) time and space.

---

## Cotangent-Weighted Mean Curvature

`vertex_mean_curvature<T>` computes discrete mean curvature at each vertex via
the Laplace-Beltrami operator:

```text
H(v_i) = || K(v_i) || / 2

K(v_i) = (1/A_i) × Σ_{j∈N(i)} (cot α_ij + cot β_ij) × (v_j − v_i)
```

where `A_i` is the barycentric area (sum of adjacent face areas / 3), and
`α_ij, β_ij` are the angles opposite edge `(i,j)` in the two sharing faces.

**Convergence**: O(h²) in edge length `h` for interior vertices on well-shaped
triangles (Meyer et al., VisMath 2003).

**Precision**: the entire computation runs in the mesh scalar `T` — no widening
from `f32` to `f64`. The gallery quality report uses `HistogramT<T>` for the
same reason.

---

## References

- Shewchuk, J.R. (1997). *Adaptive Precision Floating-Point Arithmetic and Fast
  Robust Geometric Predicates*. Discrete & Computational Geometry 18(3), 305–363.
- Ruppert, J. (1995). *A Delaunay Refinement Algorithm for Quality 2-Dimensional
  Mesh Generation*. Journal of Algorithms 18(3), 548–585.
- Jacobson, A., Kavan, L., & Sorkine-Hornung, O. (2013). *Robust Inside-Outside
  Segmentation using Generalized Winding Numbers*. ACM TOG 32(4).
- Meyer, M., Desbrun, M., Schröder, P., & Barr, A.H. (2003). *Discrete
  Differential-Geometry Operators for Triangulated 2-Manifolds*. VisMath 2002.
- van Oosterom, A. & Strackee, J. (1983). *The Solid Angle of a Plane Triangle*.
  IEEE Trans. Biomedical Engineering 30(2), 125–126.
