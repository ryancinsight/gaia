# Gaia

Gaia is the standalone repository for the `gaia` Rust crate.

`gaia` provides watertight CFD mesh generation primitives, indexed and half-edge mesh representations, exact geometric predicates, Boolean CSG operations, quality analysis, and mesh I/O. Within CFDrs, the workspace aliases `gaia` as `cfd-mesh` to preserve downstream dependency names.

## Layout

- `crates/gaia`: the standalone library crate published and consumed as `gaia`

## Features

- `parallel`: rayon-backed mesh operations
- `stl-io`, `vtk-io`, `three-mf-io`: format-specific I/O
- `millifluidic`: millifluidic-specific builders
- `cfdrs-integration`: optional bridge layer for CFDrs schematic and blueprint types

## Build

```sh
cargo check -p gaia --lib --no-default-features
cargo check -p gaia --examples --no-default-features
```

To validate the optional CFDrs bridge:

```sh
cargo check -p gaia --lib --no-default-features --features cfdrs-integration
cargo check -p gaia --examples --no-default-features --features cfdrs-integration
```
