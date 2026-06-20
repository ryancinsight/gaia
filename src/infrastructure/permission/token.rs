//! Brand-based permission token.
//!
//! The `GhostToken<'brand>` type carries a lifetime brand that ties it to a
//! specific scope. Only cells created within the same branded scope can be
//! accessed through the token.

/// A permission token branded with lifetime `'brand`.
///
/// - `&GhostToken<'brand>` grants **shared** read access to all
///   `GhostCell<'brand, T>` values.
/// - `&mut GhostToken<'brand>` grants **exclusive** write access.
///
/// The brand ensures tokens from unrelated scopes cannot be mixed.
pub struct GhostToken<'brand> {
    pub(crate) inner: melinoe::ExclusiveToken<'brand>,
}

/// A copyable, thread-safe, read-only permit for [`GhostToken`].
#[derive(Clone, Copy)]
pub struct SharedGhostToken<'a, 'brand> {
    pub(crate) inner: melinoe::SharedReadToken<'a, 'brand>,
}

impl<'brand> GhostToken<'brand> {
    /// Create a new branded token and pass it to the closure.
    ///
    /// The brand lifetime is **invariant** — it cannot be widened or narrowed —
    /// so the token cannot escape the closure or be confused with other tokens.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// GhostToken::new(|token| {
    ///     // `token` is only valid inside this closure
    /// });
    /// ```
    pub fn new<R>(f: impl for<'new_brand> FnOnce(GhostToken<'new_brand>) -> R) -> R {
        melinoe::brand_scope(|token| f(GhostToken { inner: token }))
    }

    /// Mint a `Copy`, thread-safe, read-only token tied to this borrow.
    #[inline]
    #[must_use]
    pub fn share<'a>(&'a self) -> SharedGhostToken<'a, 'brand> {
        SharedGhostToken {
            inner: self.inner.share(),
        }
    }
}

impl std::fmt::Debug for SharedGhostToken<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedGhostToken").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GhostToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GhostToken").finish_non_exhaustive()
    }
}
