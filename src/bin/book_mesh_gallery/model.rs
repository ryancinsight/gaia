use std::error::Error;

use gaia::application::watertight::check::WatertightReport;
use gaia::domain::mesh::IndexedMesh;
use hashbrown::HashSet;

pub(crate) type GalleryResult<T> = Result<T, Box<dyn Error>>;

pub(crate) struct MeshCase {
    pub(crate) slug: &'static str,
    pub(crate) title: &'static str,
    pub(crate) source: &'static str,
    pub(crate) parameters: &'static str,
    pub(crate) mesh: IndexedMesh,
}

impl MeshCase {
    /// For volume meshes: returns the set of `FaceId` indices that belong to
    /// exactly one cell (boundary faces). For surface-only meshes (no cells),
    /// returns `None` — every face is a boundary face and no filtering is needed.
    pub(crate) fn boundary_face_ids(&self) -> Option<HashSet<usize>> {
        if self.mesh.cells.is_empty() {
            return None;
        }
        let mut counts: hashbrown::HashMap<usize, u8> =
            hashbrown::HashMap::with_capacity(self.mesh.face_count());
        for cell in &self.mesh.cells {
            for &fi in &cell.faces {
                *counts.entry(fi).or_insert(0) += 1;
            }
        }
        Some(
            counts
                .into_iter()
                .filter(|&(_, c)| c == 1)
                .map(|(f, _)| f)
                .collect(),
        )
    }
}

pub(crate) struct BuildBlocker {
    pub(crate) category: &'static str,
    pub(crate) family: &'static str,
    pub(crate) source: &'static str,
    pub(crate) error: String,
}

pub(crate) struct WatertightCase {
    pub(crate) slug: &'static str,
    pub(crate) title: &'static str,
    pub(crate) source: &'static str,
    pub(crate) parameters: &'static str,
    pub(crate) mesh: IndexedMesh,
    pub(crate) report: WatertightReport,
}

pub(crate) struct WatertightRejection {
    pub(crate) slug: &'static str,
    pub(crate) title: &'static str,
    pub(crate) source: &'static str,
    pub(crate) parameters: &'static str,
    pub(crate) error: String,
}
