# ADR 0003: Align indexed CSG repair modules with their directory

- Status: Accepted
- Date: 2026-09-23
- Board item: [GAIA-013](../../backlog.md#GAIA-013)

## Context

The indexed CSG repair routines were split into files under
`indexed/repair/`, but the parent declared them with path attributes in
`indexed.rs`. Rust therefore treated the repair files as siblings of the
facade, while shared repair helpers remained in that facade. The declared
module tree did not match the directory ownership shown by the source tree.

## Decision

`indexed.rs` declares `csg` and `repair` and re-exports the two public Boolean
entry points. `indexed/csg.rs` owns operand remapping and the binary and n-ary
entry points. `indexed/repair/mod.rs` owns shared repair primitives and
declares the repair passes as child modules; `pipeline.rs` owns reconstruction
postprocessing. The existing public paths remain unchanged through the
facade re-exports. Internal cross-module access is scoped to the indexed CSG
module tree.

## Alternatives rejected

- Keep path attributes in `indexed.rs`: this preserves a module tree that
  disagrees with the filesystem and leaves repair ownership in the facade.
- Move every repair file beside `indexed.rs`: this removes the mismatch by
  discarding the cohesive repair directory.
- Broaden helper visibility to the crate: no caller outside indexed CSG needs
  these repair operations.

## Consequences

The module tree now follows the source directory hierarchy. The public CSG
entry points retain their existing paths, and repair helpers remain private
to the indexed CSG subtree. The existing Boolean tests and repair rollback
tests verify behavior across the moved boundaries.
