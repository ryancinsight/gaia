//! What to draw and how: cull mode, material, and the settings bundle.

use crate::domain::core::scalar::{Real, Vector3r};

use super::color::Rgba8;

/// Which triangles to discard before rasterising them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    /// Drop triangles whose outward normal faces away from the camera.
    ///
    /// Correct and roughly twice as fast for a closed, consistently wound
    /// surface — which is what this crate produces. Use [`Self::None`] for an
    /// open surface, or for one whose winding has not been corrected.
    #[default]
    Back,
    /// Keep every triangle; the depth test resolves visibility on its own.
    None,
}

/// A flat-shaded surface material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// The unlit surface colour.
    pub base_color: Rgba8,
    /// Fraction of the base colour present with no light contribution.
    pub ambient: Real,
    /// Fraction of the base colour that the diffuse term may add.
    pub diffuse: Real,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Rgba8::rgb(180, 190, 200),
            ambient: 0.25,
            diffuse: 0.75,
        }
    }
}

impl Material {
    /// The default neutral surface.
    #[must_use]
    pub fn surface() -> Self {
        Self::default()
    }

    /// Builder form of `base_color`.
    #[must_use]
    pub const fn with_base_color(mut self, base_color: Rgba8) -> Self {
        self.base_color = base_color;
        self
    }

    /// The flat colour for a triangle with unit outward `normal`, lit by a
    /// headlight along `light_dir`.
    ///
    /// The Lambert term is `|n·l|`: an absolute value, so a triangle facing away
    /// from the light is shaded as if it faced towards it. That is deliberate.
    /// A viewer is often handed a mesh whose winding is not yet consistent, and
    /// `max(0, n·l)` would render half of such a mesh black and make the winding
    /// look like a lighting bug.
    #[must_use]
    pub fn shade(&self, normal: &Vector3r, light_dir: &Vector3r) -> Rgba8 {
        let lambert = normal.dot(*light_dir).abs().clamp(0.0, 1.0);
        let intensity = (self.ambient + self.diffuse * lambert).clamp(0.0, 1.0);
        self.base_color.scaled(intensity)
    }
}

/// What to draw and how.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    /// Colour written where no triangle covers a pixel.
    pub background: Rgba8,
    /// The surface material.
    pub material: Material,
    /// Which triangles to discard.
    pub cull: CullMode,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            background: Rgba8::DARK_GRAY,
            material: Material::surface(),
            cull: CullMode::Back,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_uses_an_ambient_floor_and_an_absolute_lambert_term() {
        let m = Material {
            base_color: Rgba8::rgb(200, 200, 200),
            ambient: 0.25,
            diffuse: 0.75,
        };
        let light = Vector3r::new(1.0, 0.0, 0.0);
        let facing = m.shade(&Vector3r::new(1.0, 0.0, 0.0), &light);
        let away = m.shade(&Vector3r::new(-1.0, 0.0, 0.0), &light);
        let edge_on = m.shade(&Vector3r::new(0.0, 0.0, 1.0), &light);
        assert_eq!(facing.r, 200);
        assert_eq!(away, facing, "the Lambert term is absolute");
        assert_eq!(edge_on.r, 50, "ambient only at 90 degrees");
    }
}
