//! Connected-component pruning: keep the largest shell, drop phantom islands.

use super::IndexedMesh;
use crate::domain::core::index::{FaceId, RegionId, VertexId};
use crate::domain::core::scalar::Scalar;

impl<T: Scalar> IndexedMesh<T> {
    /// Remove all connected face components except the largest one.
    ///
    /// After a CSG Difference operation, phantom closed "islands" — small groups
    /// of faces that are locally manifold but disconnected from the main body —
    /// can appear at the seam boundary.  Each island passes `is_watertight = true`
    /// individually but inflates the Euler characteristic (χ = 4 instead of 2),
    /// corrupts the signed volume, and produces floating geometry in STL output.
    ///
    /// **Threshold:** a component is discarded when its face count is less than
    /// `max(4, largest_component_size × 5%)`.  The 5 % relative threshold with
    /// a 4-face absolute minimum reliably suppresses seam artefacts while
    /// preserving large intentional secondary bodies.
    ///
    /// **Field handling after reconstruction:**
    ///
    /// | Field             | Action                                      |
    /// |-------------------|---------------------------------------------|
    /// | `vertices`        | Rebuilt — orphaned vertices removed          |
    /// | `faces`           | Rebuilt from kept-component faces only       |
    /// | `edges`           | Set to `None` — lazily rebuilt on next call  |
    /// | `cells`           | Cleared — CSG surfaces carry no cell data    |
    /// | `attributes`      | Remapped via old→new `FaceId` translation    |
    /// | `boundary_labels` | Remapped; labels on discarded faces dropped  |
    ///
    /// # Returns
    /// Number of components discarded (`0` when the mesh was already clean).
    pub fn retain_largest_component(&mut self) -> usize {
        use crate::domain::topology::connectivity::connected_components;
        use crate::domain::topology::AdjacencyGraph;

        // Ensure edge adjacency is current.
        self.rebuild_edges();
        let edges = self.edges.as_ref().expect("edges just rebuilt");

        let adj = AdjacencyGraph::build(&self.faces, edges);
        let components = connected_components(&self.faces, &adj);

        // Fast path: already a single component.
        if components.len() <= 1 {
            return 0;
        }

        let largest_size = components.iter().map(std::vec::Vec::len).max().unwrap_or(0);
        // Discard if face_count < max(4, largest * 0.05).
        let min_keep = ((largest_size as f64 * 0.05).ceil() as usize).max(4);

        let mut discarded = 0;
        for component in &components {
            if component.len() < min_keep {
                discarded += 1;
            }
        }

        // Avoid allocating a reconstruction mesh and two remap arrays when
        // every component already satisfies the retention threshold.
        if discarded == 0 {
            return 0;
        }

        // Single-pass over components: build a fresh mesh from kept faces only.
        let mut new_mesh = self.empty_clone();
        // Valid mesh IDs are always below `u32::MAX`, so the maximum raw ID
        // is a compact sentinel replacement for `Option<VertexId>` and
        // `Option<FaceId>`.
        const UNMAPPED: u32 = u32::MAX;
        let mut vertex_remap = vec![UNMAPPED; self.vertices.len()];
        let mut face_remap = vec![UNMAPPED; self.faces.len()];

        for component in &components {
            if component.len() < min_keep {
                tracing::debug!(
                    "retain_largest_component: discarding {} phantom face(s) \
                      (threshold = {} faces)",
                    component.len(),
                    min_keep,
                );
                continue;
            }
            for &old_fid in component {
                let fd = *self.faces.get(old_fid);
                let mut nv = [VertexId::default(); 3];
                for (k, &vid) in fd.vertices.iter().enumerate() {
                    let idx = vid.as_usize();
                    nv[k] = if vertex_remap[idx] == UNMAPPED {
                        let new_vid = new_mesh
                            .add_vertex(*self.vertices.position(vid), *self.vertices.normal(vid));
                        vertex_remap[idx] = new_vid.raw();
                        new_vid
                    } else {
                        VertexId::new(vertex_remap[idx])
                    };
                }
                // Guard: skip any face that collapsed under vertex welding.
                if nv[0] == nv[1] || nv[1] == nv[2] || nv[2] == nv[0] {
                    continue;
                }
                let new_fid = if fd.region == RegionId::INVALID {
                    new_mesh.add_face(nv[0], nv[1], nv[2])
                } else {
                    new_mesh.add_face_with_region(nv[0], nv[1], nv[2], fd.region)
                };
                face_remap[old_fid.as_usize()] = new_fid.0;
            }
        }

        // Remap per-face scalar attributes.
        let old_attrs = std::mem::take(&mut self.attributes);
        for channel in old_attrs.channel_names() {
            for (old_fid_idx, &opt_new_fid) in face_remap.iter().enumerate() {
                if opt_new_fid != UNMAPPED {
                    let old_fid = FaceId::from_usize(old_fid_idx);
                    if let Some(val) = old_attrs.get(channel, old_fid) {
                        new_mesh.attributes.set(channel, FaceId(opt_new_fid), val);
                    }
                }
            }
        }

        // Remap boundary labels.
        let old_labels = std::mem::take(&mut self.boundary_labels);
        new_mesh.boundary_labels = old_labels
            .into_iter()
            .filter_map(|(old_fid, label)| {
                let new_fid = face_remap[old_fid.as_usize()];
                (new_fid != UNMAPPED).then_some((FaceId(new_fid), label))
            })
            .collect();

        // Swap stores in-place.
        self.vertices = new_mesh.vertices;
        self.faces = new_mesh.faces;
        self.edges = None; // stale; lazily rebuilt on next use
        self.cells = Vec::new(); // CSG surfaces carry no volumetric cells
        self.attributes = new_mesh.attributes;
        self.boundary_labels = new_mesh.boundary_labels;

        tracing::debug!(
            "retain_largest_component: removed {} component(s); {} faces remain",
            discarded,
            self.faces.len(),
        );
        discarded
    }
}
