# cfd-mesh — Agent Reference

> **Role**: Standalone mesh geometry/topology crate with half-edge topology, welding,
> watertight validation, CSG, and mesh I/O.
> **Optional integration**: `cfdrs-integration` enables the CFDrs schematic and
> blueprint bridge without making it part of the default crate surface.

---

## Current Snapshot

- Crate name: `cfd-mesh`
- Public top-level modules: `application`, `domain`, `infrastructure`
- Core public types include `IndexedMesh`, `MeshBuilder`, `BoundaryPatch`, `PatchType`,
  analytic primitive builders, and quality-analysis helpers.
- The standalone default build does not require `cfd-schematics`.
- Blueprint and scheme bridge entry points are gated behind `feature = "cfdrs-integration"`.

## Module Layout

```text
src/
  lib.rs                      public re-exports
  application/                channel builders, CSG, welding, quality, watertight, pipeline
  domain/                     core types, geometry, mesh, topology
  infrastructure/             I/O, permission, spatial, storage
  bin/                        utility binaries
```

## Feature Flags

- `csg`: enables Boolean / CSG operations and related examples
- `millifluidic`: enables millifluidic-specific builders and examples
- `parallel`: enables rayon-backed parallel operations
- `stl-io`, `vtk-io`, `three-mf-io`: format-specific I/O support
- `scheme-io`: compatibility alias for the CFDrs bridge
- `cfdrs-integration`: enables blueprint pipeline, shell pipeline, and scheme import

## Example Gating Rule

- `autoexamples = false` must remain enabled in `Cargo.toml`.
- Every example file must be declared explicitly.
- Examples that require CFDrs blueprint or scheme types must set
  `required-features = ["cfdrs-integration"]`.

## Validation Commands

```sh
cargo check -p cfd-mesh --lib --no-default-features
cargo check -p cfd-mesh --examples --no-default-features
cargo check -p cfd-mesh --example schematic_to_3d_mesh --no-default-features
cargo check -p cfd-mesh --lib --no-default-features --features cfdrs-integration
cargo check -p cfd-mesh --examples --no-default-features --features cfdrs-integration
```

Expected behavior:

- standalone builds succeed without `cfdrs-integration`
- gated blueprint examples fail cleanly when the feature is absent
- full example and library builds succeed when the feature is enabled
