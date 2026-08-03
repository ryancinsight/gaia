# ADR-0001: Validate channel paths and variable sweeps

- Status: Accepted
- Date: 2026-08-02
- Board item: [Phase 48](../../CHECKLIST.md#phase-48-mesh-builder-input-and-branch-stability-safety)

## Context

`ChannelPath::new` accepted invalid input and asserted when fewer than two
points were supplied. `SweepMesher::sweep_variable` silently returned an empty
face list when its width-scale count did not match the path station count.
Both behaviors made caller mistakes look like valid mesh results or process
panics. The path also retained a growable `Vec` after construction even though
its waypoint sequence is immutable.

## Decision

`ChannelPath::new` and `ChannelPath::straight` return
`Result<ChannelPath, ChannelPathError>`. Construction rejects too few points,
non-finite coordinates, and zero-length segments. Validated waypoints are
stored as `Box<[Point3r]>`, so capacity is frozen at construction. The public
`segment_direction` query returns `Option<Vector3r>` for an out-of-range
segment.

`SweepMesher::sweep_variable` returns `Result<Vec<FaceData>, SweepError>` and
reports an exact expected/actual station-count mismatch without touching the
vertex pool. It uses the validated path length for the count check, avoiding a
temporary frame allocation before the canonical sweep kernel computes frames.

## Alternatives rejected

- Keep the assertion and document the panic: this leaves hostile or malformed
  input on a public input-dependent panic path.
- Return an empty face list on mismatch: this is silent failure and can produce
  an apparently valid empty downstream mesh.
- Add a downstream adapter: the invariant belongs at Gaia's canonical channel
  boundary and must be shared by Atlas consumers.

## Consequences

This is a breaking public API change. In-repository callers use `?` or explicit
typed handling; external callers follow the [migration guide](../migration/channel-path-validation.md).
The branching Boolean composition remains a separate open stability item in
Phase 48 because its `NotWatertight` failures require a topology-level repair,
not an input-error substitution.

## Evidence

- `cargo check --lib --offline`
- `cargo check --bin book_mesh_gallery --offline`
- Focused nextest: `channel::path::tests` and `channel::sweep::tests`, 4/4
- The gallery executes from the current Gaia library artifact and records
  exact mesh counts in `docs/book/figure_manifest.md`.
