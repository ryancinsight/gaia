# Channel path and variable sweep migration

The channel input contract is now fallible and typed.

## Channel paths

Handle construction errors instead of relying on a panic:

```rust
use gaia::{ChannelPath, ChannelPathError};

let path = ChannelPath::new(points)?;
```

`ChannelPath::straight(start, end)` also returns
`Result<ChannelPath, ChannelPathError>`. The constructor rejects fewer than
two points, non-finite coordinates, and adjacent duplicate points.

`segment_direction(index)` returns `Option<Vector3r>`; `None` means the index
does not identify a segment.

## Variable-width sweeps

Handle a scale-count mismatch explicitly:

```rust
use gaia::{SweepError, SweepMesher};

let faces = SweepMesher::new().sweep_variable(
    &profile,
    &path,
    &width_scales,
    &mut vertex_pool,
    region,
)?;
```

The error is `SweepError::WidthScaleCountMismatch { expected, actual }`. The
vertex pool is unchanged when the counts differ.
