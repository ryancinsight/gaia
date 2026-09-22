//! The colour the rasteriser writes, and how it packs into a host word.

use crate::domain::core::scalar::Real;

/// An 8-bit-per-channel colour.
///
/// [`Self::packed`] produces `0xAARRGGBB`, which is the format the host
/// framebuffer expects, so a rendered frame can be copied across without a
/// swizzle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rgba8 {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel.
    pub a: u8,
}

impl Rgba8 {
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    /// Mid grey.
    pub const GRAY: Self = Self::rgb(128, 128, 128);
    /// Dark grey, the default viewport background.
    pub const DARK_GRAY: Self = Self::rgb(32, 34, 38);
    /// Light grey.
    pub const LIGHT_GRAY: Self = Self::rgb(200, 200, 200);

    /// Create a colour from its four channels.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque colour from three channels.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// Pack into the host framebuffer's `0xAARRGGBB` word.
    #[must_use]
    pub const fn packed(self) -> u32 {
        ((self.a as u32) << 24) | ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
    }

    /// Scale the colour channels by `factor`, clamping into range.
    ///
    /// Alpha is left alone: scaling coverage is not what a lighting model
    /// means, and a partially transparent surface is the host's decision. A
    /// `NaN` factor yields zero rather than propagating a `NaN` into a channel
    /// cast, which would be an unspecified value.
    #[must_use]
    pub fn scaled(self, factor: Real) -> Self {
        let scale = |channel: u8| -> u8 {
            let value = Real::from(channel) * factor;
            if value.is_nan() || value <= 0.0 {
                0
            } else if value >= 255.0 {
                255
            } else {
                value as u8
            }
        };
        Self {
            r: scale(self.r),
            g: scale(self.g),
            b: scale(self.b),
            a: self.a,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_matches_the_host_framebuffer_word_order() {
        // 0xAARRGGBB, the same value `u32::from_be_bytes([a, r, g, b])` gives.
        assert_eq!(Rgba8::new(0x11, 0x22, 0x33, 0x44).packed(), 0x4411_2233);
        assert_eq!(Rgba8::WHITE.packed(), 0xFFFF_FFFF);
        assert_eq!(Rgba8::BLACK.packed(), 0xFF00_0000);
    }

    #[test]
    fn scaled_clamps_and_leaves_alpha_alone() {
        let c = Rgba8::new(100, 200, 250, 128);
        assert_eq!(c.scaled(0.0), Rgba8::new(0, 0, 0, 128));
        assert_eq!(c.scaled(1.0), c);
        assert_eq!(c.scaled(10.0), Rgba8::new(255, 255, 255, 128));
        assert_eq!(c.scaled(-1.0), Rgba8::new(0, 0, 0, 128));
        assert_eq!(c.scaled(f64::NAN), Rgba8::new(0, 0, 0, 128));
    }
}
