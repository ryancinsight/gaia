# ADR 0002: Separate boundary-facet and volume-cell quality criteria

- Status: Accepted
- Date: 2026-08-04
- Driver: `CHECKLIST.md` Phase 52

## Context

Quality tetrahedral meshing has two distinct acceptance surfaces. A volume
cell can satisfy radius-edge, dihedral-angle, normalized-volume, and volume
bounds while an exposed triangular facet is too large or poorly shaped. The
primary CGAL Mesh_3 contract models facet and cell criteria separately. TetGen
also exposes facet-area and boundary-segment sizing alongside tetrahedral
quality and volume constraints.

Gaia already owns native-precision tetrahedral cell metrics and explicit cell
criteria. It did not yet provide an executable boundary-facet policy or a
boundary-cell result that combined the two policies.

## Decision

Gaia adds `BoundaryFacetQualityCriteria<T>` with explicit native-precision
bounds for minimum facet angle, shortest-to-longest edge ratio, and optional
maximum edge length. `TetrahedralQualityCriteria<T>::assess_boundary` composes
that policy with the cell policy.

Boundary facets are identified by face incidence: exactly one tetrahedral cell
references the face. Each geometric boundary facet is measured once. A
boundary cell passes only when its cell criteria and every exposed facet pass.
Malformed tetrahedral topology or facet geometry is rejected and counted as
invalid; it is never silently treated as an interior cell or assigned a
default metric.

This increment defines acceptance only. Feature protection, sizing fields,
constrained Delaunay refinement, termination limits, and sliver optimization
remain separate follow-up capabilities and require a consumer workload and
controlled performance evidence.

## Alternatives rejected

- A single hidden default facet threshold: rejected because quality and sizing
  are consumer policy and differ across CFD, FEM, and geometry consumers.
- Reusing the all-cell report as a boundary contract: rejected because it
  cannot distinguish exposed facets from interior faces or enforce facet size.
- Starting constrained refinement before an acceptance oracle: rejected because
  a refinement loop without facet, cell, and termination criteria cannot be
  verified for correctness or convergence.

## Verification

The boundary module tests native `f32` and `f64` instantiations, accepted unit
tetrahedra, oversized facets, malformed facet vertex identifiers, malformed
cell topology, and invalid criterion domains. The relevant primary references
are linked from [`docs/mesh_library_gap_audit.md`](../mesh_library_gap_audit.md).
