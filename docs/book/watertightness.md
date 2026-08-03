# Watertightness diagnostics

Gaia treats watertightness as a publish boundary for surface meshes. The
canonical `check_watertight` report records closure, boundary-edge count,
non-manifold-edge count, orientation consistency, signed volume, and Euler
characteristic. The report is evidence about a concrete `IndexedMesh`; it is
not inferred from a builder name or from a successful `Result`.

![Watertightness diagnostic cases](figures/watertightness-diagnostics.svg)

The generated sheet contains four analytical diagnostics:

- the closed cube is the positive closed-manifold reference;
- removing one triangular face exposes three red boundary edges;
- duplicating one face creates three orange non-manifold edges;
- reversing one face preserves closure but fails orientation consistency.

The final panels are the deterministic branching representatives. The public
builder returns a typed error for each because the Boolean result is not
watertight. The generator intentionally renders a rejection panel instead of
the invalid result. The current counts are `NotWatertight { count: 4 }` for
bifurcation and `NotWatertight { count: 15 }` for trifurcation. A separate
postcondition also rejects a topologically closed result that omits an outlet
region or its analytical outlet neighborhood.

The exact report values, source modules, parameters, and rejection text are in
the [watertightness figure manifest](watertightness_manifest.md). The red and
orange edge classifications are derived from Gaia's persistent `EdgeStore`;
they are not manually marked in the SVG.

## Review rule

The diagnostic SVG is rasterized during verification and inspected for
off-canvas geometry, clipped labels, missing failure highlights, and accidental
rendering of rejected branch geometry. A passing link check or mdBook build is
not a substitute for this visual review.
