//! `GhostCell` — zero-cost interior mutability gated by a branded token.
//!
//! The `GhostCell<'brand, T>` wrapper stores data that can only be accessed
//! through a matching `GhostToken<'brand>`.

use super::token::{GhostToken, SharedGhostToken};
use melinoe::MelinoeCell;

/// A cell whose contents are accessible only via a matching [`GhostToken`].
///
/// This is safe because:
/// - `&GhostToken` → `&T` (shared borrow of token implies shared borrow of data)
/// - `&mut GhostToken` → `&mut T` (exclusive borrow of token implies exclusive borrow of data)
///
/// The Rust borrow checker enforces that you cannot hold `&T` and `&mut T` simultaneously
/// because that would require `&GhostToken` and `&mut GhostToken` at the same time.
pub struct GhostCell<'brand, T: ?Sized> {
    pub(crate) inner: MelinoeCell<'brand, T>,
}

// SAFETY: GhostCell is Send/Sync when T is Send/Sync because access is
// gated by the single-threaded token discipline.
// The token itself is !Send + !Sync, so cross-thread access is prevented.
//
// However: for our mesh use-case, we want the *data* to be shareable across threads
// while the token stays on one thread. This is safe because the token
// prevents simultaneous mutable access.
// We mark Send+Sync only when T: Send+Sync.

// SAFETY: Access to the inner value is gated by GhostToken which ensures
// exclusive mutable access through the borrow checker. When T: Send,
// ownership transfer across threads is safe because, at any point,
// only one thread can hold &mut GhostToken.
unsafe impl<T: ?Sized + Send> Send for GhostCell<'_, T> {}

// SAFETY: Shared references (&GhostCell) can exist on multiple threads,
// but reading requires &GhostToken (which is !Sync by construction of the
// invariant lifetime brand in PhantomData). Writing requires
// &mut GhostToken. The borrow checker prevents data races.
unsafe impl<T: ?Sized + Send + Sync> Sync for GhostCell<'_, T> {}

impl<'brand, T> GhostCell<'brand, T> {
    /// Wrap a value in a `GhostCell`.
    #[inline]
    pub const fn new(value: T) -> Self {
        GhostCell {
            inner: MelinoeCell::new(value),
        }
    }

    /// Borrow the contents immutably. Requires `&GhostToken<'brand>`.
    #[inline]
    pub fn borrow<'a>(&'a self, token: &'a GhostToken<'brand>) -> &'a T {
        self.inner.borrow(&token.inner).into_ref()
    }

    /// Borrow the contents immutably using a shared read token.
    #[inline]
    pub fn borrow_shared<'a>(&'a self, token: SharedGhostToken<'a, 'brand>) -> &'a T {
        self.inner.borrow(token.inner).into_ref()
    }

    /// Borrow the contents mutably. Requires `&mut GhostToken<'brand>`.
    #[inline]
    pub fn borrow_mut<'a>(&'a self, token: &'a mut GhostToken<'brand>) -> &'a mut T {
        self.inner.borrow_mut(&mut token.inner).into_mut()
    }

    /// Consume the cell, returning the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

impl<'brand, T: Clone> GhostCell<'brand, T> {
    /// Clone the inner value. Requires `&GhostToken`.
    #[inline]
    pub fn clone_inner(&self, token: &GhostToken<'brand>) -> T {
        self.borrow(token).clone()
    }
}

impl<T: Default> Default for GhostCell<'_, T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<'brand, T: std::fmt::Debug> GhostCell<'brand, T> {
    /// Debug-format the inner value. Requires `&GhostToken`.
    pub fn debug_with(
        &self,
        token: &GhostToken<'brand>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        self.borrow(token).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `GhostCell<T: Send>` can be sent to another thread.
    ///
    /// # Theorem — Send safety
    ///
    /// `unsafe impl Send for GhostCell<'_, T> where T: Send` is sound because:
    /// - `GhostToken` is `!Send` (invariant brand lifetime + non-`Send` inner).
    ///   It cannot migrate between threads.
    /// - Moving a `GhostCell` to another thread transfers *ownership* of `T`.
    ///   The receiver has no matching token, so it cannot read or write through
    ///   the cell — it can only hold it until ownership returns to the
    ///   token-holding thread.
    /// - No data race is possible: the moved cell is inaccessible without
    ///   the token, which stays on the original thread.
    ///
    /// Uses `thread::scope` so the brand lifetime `'brand` does not need to
    /// be `'static` — the scope is tied to the outer frame where the token
    /// lives.
    #[test]
    fn ghost_cell_send_owned_across_thread() {
        GhostToken::scope(|mut token| {
            let cell = GhostCell::new(String::from("hello"));
            *cell.borrow_mut(&mut token) = String::from("world");

            // thread::scope lets us move the cell to a worker without
            // requiring 'static: the scope's lifetime is bounded by this
            // closure, which is also bounded by the token's brand lifetime.
            let result = std::thread::scope(|s| {
                s.spawn(|| {
                    // No token available on this thread — can only consume.
                    cell.into_inner()
                })
                .join()
                .expect("worker thread panicked")
            });

            assert_eq!(result, "world");
        });
    }

    /// A shared reference `&GhostCell<T: Send + Sync>` can exist on multiple
    /// threads simultaneously — exercised via `thread::scope`.
    ///
    /// # Theorem — Sync safety
    ///
    /// `unsafe impl Sync for GhostCell<'_, T> where T: Send + Sync` is sound
    /// because:
    /// - `&GhostCell` alone grants no read access to `T`.  Any read requires
    ///   `&GhostToken` and any write requires `&mut GhostToken`.
    /// - `GhostToken` is `!Sync`, so only the token-owning thread can produce
    ///   a `&GhostToken` for reads.
    /// - Concurrent `&GhostCell` holders on other threads hold an opaque
    ///   reference and cannot observe `T` at all.
    /// - The borrow-checker enforces that `&GhostToken` and `&mut GhostToken`
    ///   cannot coexist, preventing data races.
    #[test]
    fn ghost_cell_sync_shared_ref_across_threads() {
        GhostToken::scope(|token| {
            let cell = GhostCell::new(42_u64);

            // The worker holds &cell (a shared reference) across the scope.
            // It cannot read the value — no token is passed to it.
            let worker_saw_ref = std::thread::scope(|s| {
                s.spawn(|| {
                    // Hold a reference to the cell without reading its value.
                    // This exercises the Sync impl: &GhostCell is shareable.
                    core::ptr::from_ref(&cell).addr()
                })
                .join()
                .expect("worker thread panicked")
            });

            // Main thread reads with its token while/after the worker runs.
            let val = *cell.borrow(&token);
            assert_eq!(val, 42_u64);
            // Verify the worker actually held the reference (non-null address).
            assert_ne!(worker_saw_ref, 0, "worker held a null cell reference");
        });
    }
}
