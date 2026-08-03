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

“All possible meshes” is finite only at the family level. Each family accepts
continuous dimensions, resolutions, profiles, or fields, so the gallery uses
small deterministic representative parameters and records them in the
generated manifest. A new public mesh-producing family is a documentation
change: add it to the generator, regenerate the figures, review them, and
update this coverage list in the same change.

## Evidence boundary

The generated SVGs are display projections. The manifest preserves the source
builder name and mesh counts; it also records the display sampling limit used
to keep the figures inspectable. The figures are reviewed as rendered PNGs,
but the raster review does not replace numerical mesh validation or topology
tests.

The gallery does not hide the current Boolean stability findings. A
binary union/difference probe used during the audit returned
`NotWatertight { count: 12 }` from the branch-connection path. The gallery uses
the public one-operand n-ary identity only to keep the CSG family represented
by a real Gaia output; binary Boolean repair remains an explicit audit item.

A repeatability audit found unordered intermediate traversal in the
tetrahedral and branching paths. Before the fix, re-running the same gallery
inputs changed the SDF volume representative from `V=158 F=1016 C=430` to
`V=158 F=1062 C=453`, and changed the branching errors from counts `4/15` to
`7/14`. The lattice macro-block order, broad-phase candidate stream, CSG
arrangement emission, and repair traversal are now canonicalized. The focused
regression compares repeated same-process branching errors, and two
consecutive fresh gallery runs have identical manifest and topology SVG
SHA-256 hashes. The current stable values are SDF `V=158 F=1000 C=422` and
branching error counts `4/15`.

The same failure is currently returned by the public bifurcation and
trifurcation channel builders because they compose their branch geometry with
the n-ary Boolean path. Their representative parameters and exact errors are
kept in the generated manifest; the channel sheet contains only builders that
produced real `IndexedMesh` values.
