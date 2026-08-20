# Executable quick start

Gaia's primitive builders return a real indexed surface mesh. The smallest
closed-solid example uses a centred cube and validates both the generated
topology and its expected triangular-face count:

```rust
extern crate gaia;

use gaia::domain::geometry::primitives::{Cube, PrimitiveMesh};

fn main() {
    let mut mesh = Cube::centred(2.0).build().expect("unit cube construction");

    assert!(mesh.is_watertight());
    assert_eq!(mesh.vertices.len(), 8);
    assert_eq!(mesh.faces.len(), 12);
}
```

The assertions are value-semantic checks over the returned mesh: the builder
must emit eight deduplicated corner vertices, twelve triangles, and a closed
surface.
