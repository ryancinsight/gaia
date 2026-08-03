use std::error::Error;

use gaia::domain::mesh::IndexedMesh;

pub(crate) type GalleryResult<T> = Result<T, Box<dyn Error>>;

pub(crate) struct MeshCase {
    pub(crate) slug: &'static str,
    pub(crate) title: &'static str,
    pub(crate) source: &'static str,
    pub(crate) parameters: &'static str,
    pub(crate) mesh: IndexedMesh,
}

pub(crate) struct BuildBlocker {
    pub(crate) category: &'static str,
    pub(crate) family: &'static str,
    pub(crate) source: &'static str,
    pub(crate) error: String,
}
