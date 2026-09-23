# ADR 0004: Derive symmetric GWN classification thresholds

- Status: Accepted
- Date: 2026-09-23
- Board item: GAIA-001 ([PR #70](https://github.com/ryancinsight/gaia/pull/70))

## Context

`GWN_OUTSIDE_THRESHOLD` and `GWN_INSIDE_THRESHOLD` were 0.35 and 0.65 without
a derivation. The backlog asked for a misclassification probability or a
scale-aware criterion, but the generalized winding number is dimensionless and
the repository has no input-error distribution from which to derive a
probability.

For a consistently oriented watertight solid, the exact winding number is 0
outside and signed 1 inside, away from the boundary. Across a planar boundary,
the one-sided limits differ by one. Open, non-manifold, and duplicated faces
can instead produce a confidence field with shifted values.

Reference: Jacobson, Kavan, and Sorkine-Hornung, “Robust Inside-Outside
Segmentation using Generalized Winding Numbers,” SIGGRAPH 2013, §§4.1–4.2
([paper](https://igl.ethz.ch/projects/winding-number/robust-inside-outside-segmentation-using-generalized-winding-numbers-siggraph-2013-compressed-jacobson-et-al.pdf)).
Section 4.1 defines the solid-angle winding number and its closed-surface
classification; §4.2 describes open, non-manifold, and duplicated-surface
behavior.

## Decision

Use the symmetric thresholds `t` and `1 − t`. The reference magnitudes are
0 outside, 1 inside, and 0.5 at the midpoint of the boundary’s one-sided
limits. The minimum additive margin to the two decision thresholds is
`m(t) = min(t, 0.5 − t)`. Equalizing the terms gives the unique maximum at
`t = 0.25`, so the constants are 0.25 and 0.75.

The criterion is scale-invariant because winding number is dimensionless. The
values provide a deterministic maximin margin for those reference magnitudes;
they do not bound floating-point error or misclassification probability and
do not guarantee correct thresholding for arbitrary triangle soups. Values in
the band continue through the existing refinement and geometric tiebreakers.
The solid-angle clamp is a separate per-face numerical policy and does not
derive the classification thresholds.

## Alternatives rejected

- Keep 0.35 and 0.65: their minimum margin to 0, 0.5, and 1 is 0.15, below
  the 0.25 maximum.
- Scale the thresholds by mesh dimensions: uniform scaling leaves winding
  number unchanged and supplies no relevant scale parameter.
- Claim a misclassification probability: no distribution over input meshes,
  geometry, or arithmetic error is defined.

## Consequences

The constants remain source-compatible, with values 0.25 and 0.75. Tests pin
the derivation and exercise band classification against exact orientation on
an open face. A closed-cube face-center query also reaches the band and is
resolved by the coplanar tiebreaker for either fragment orientation. Closed-
solid inside and outside cases exercise the two decision regions. Consumers
must treat the thresholds as a decision policy, not a probability or universal
guarantee.
