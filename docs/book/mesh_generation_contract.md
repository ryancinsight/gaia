# Mesh-generation contract

Gaia owns mesh construction in Atlas. The canonical output is
`gaia::IndexedMesh`, which stores deduplicated vertices, indexed triangular
faces, persistent edge topology, optional boundary labels, and volumetric
cells. Consumers should receive this value or a typed boundary conversion;
they should not clone Gaia's meshing algorithms in application repositories.

The gallery covers every public mesh-producing family currently enumerated by
Gaia's primitive module and its public application builders:

- analytic primitives, including the TPMS sphere families;
- channel, sweep, branching, venturi, serpentine, and substrate builders;
- structured tetrahedral and hexahedral grids;
- hexahedron-to-tetrahedron conversion and P2 surface refinement;
- the public one-operand CSG n-ary identity contract and SDF-driven
  tetrahedral volume generation.

Channel paths are validated at construction and store frozen waypoint
capacity. Variable-width sweeps report a typed station-count mismatch without
mutating their vertex pool. These contracts keep malformed input out of the
mesh kernel; the public API migration is recorded in
`docs/migration/channel-path-validation.md`.

Branching builders reject non-finite or non-positive dimensions, angles outside
`(0, π/2)`, unsupported daughter counts, undersized resolutions, and capacity
overflow before constructing tube meshes. Tube construction reserves its
topology-derived storage and retains only the first and previous axial rings;
this bounds temporary ring-index storage by angular resolution rather than
axial resolution without changing the emitted faces.

Branching Boolean output has a second safety boundary: a watertight result must
retain every daughter outlet region and a vertex neighborhood around each
analytical outlet center. A result that is topologically watertight but omits a
daughter is rejected with a typed `BuildError`; the focused regression covers
this false-positive class without publishing approximate geometry.

The [watertightness diagnostics](watertightness.md) figure exercises the
canonical report against a closed cube, a removed-face boundary, a duplicated
face non-manifold edge, and an inconsistent face winding. Analytical failure
cases remain visibly marked with their boundary, non-manifold, or orientation
diagnostics. Branch representatives are rendered only after the builder's
watertightness and daughter-outlet postconditions pass; a `NotWatertight` or
missing-outlet result is never rendered as a valid channel.

“All possible meshes” is finite only at the family level. Each family accepts
continuous dimensions, resolutions, profiles, or fields, so the gallery uses
small deterministic representative parameters and records them in the
generated manifest. A new public mesh-producing family is a documentation
change: add it to the generator, regenerate the figures, review them, and
update this coverage list in the same change.

Quality is provider-owned as well. Surface validation evaluates triangle
metrics in the mesh scalar, CFD volume-cell reports expose internal-face
non-orthogonality and skewness, and tetrahedral reports expose native
volume, radius-edge ratio, minimum dihedral angle, and normalized volume.
Invalid tetrahedral cells are counted rather than assigned default metric
values. `TetrahedralQualityCriteria<T>` adds an explicit acceptance boundary:
callers provide the shape bounds and may provide a maximum cell volume. The
validated policy classifies cells as accepted, sliver candidates, poor-shape,
oversized, or invalid. There is no default because mesh quality and sizing
are consumer policy. `BoundaryFacetQualityCriteria<T>` separately validates
minimum facet angle, edge-length ratio, and optional maximum edge length.
`TetrahedralQualityCriteria<T>::assess_boundary` identifies geometric boundary
facets by single-cell incidence and accepts a boundary cell only when both its
cell policy and all exposed facet policies pass. Malformed cell or facet
topology is rejected and counted as invalid. Constrained three-dimensional
refinement, feature protection, sizing fields, and sliver optimization are
not implied by these acceptance measurements and remain separate capabilities.

## Evidence boundary

The generated SVGs are display projections. The manifest preserves the source
builder name and mesh counts; it also records the display sampling limit used
to keep the figures inspectable. The figures are reviewed as rendered PNGs,
but the raster review does not replace numerical mesh validation or topology
tests.

The gallery exercises the representative branch-connection path through the
real n-ary Boolean union. Operands outside Gaia's stable `0.5..=10.0` combined
AABB-diagonal band are translated and scaled to a unit diagonal before
numerical predicates, then mapped back afterward. Operands inside that band
retain their original coordinates and are borrowed rather than cloned, so
exact decimal coplanarity in common axis-aligned solids is preserved without
an extra operand allocation. Only the out-of-band path owns transformed
operand storage. Tube walls use outward winding, and small circumferential
edges select a smaller local snap cell so they cannot weld together. The
resulting bifurcation and trifurcation meshes are watertight and retain every
daughter outlet region.

A repeatability audit found unordered intermediate traversal in the
tetrahedral and branching paths. Before the fix, re-running the same gallery
inputs changed the SDF volume representative from `V=158 F=1016 C=430` to
`V=158 F=1062 C=453`, and changed the branching errors from counts `4/15` to
`7/14`. The lattice macro-block order, broad-phase candidate stream, CSG
arrangement emission, and repair traversal are now canonicalized. The focused
regression compares repeated same-process branching results, and two
consecutive fresh gallery runs have identical manifest and topology SVG
SHA-256 hashes. The current stable values are SDF `V=158 F=1000 C=422`,
bifurcation `V=414 F=824`, and trifurcation `V=909 F=1814`.

The public bifurcation and trifurcation channel builders now compose their
branch geometry with the n-ary Boolean path and publish real `IndexedMesh`
values. Their representative parameters and exact reports are kept in the
generated manifests; the channel sheet contains both branch families.
