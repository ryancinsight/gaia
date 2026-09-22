//! Contiguous rows addressed by an offset table — the stack's one CSR shape.
//!
//! A relation that maps an entity id to a variable-length list of items is
//! stored here as **one offset table plus one contiguous value buffer**:
//! row `r` occupies `values[offsets[r]..offsets[r + 1]]`. Rows are therefore
//! a slice walk rather than a pointer chase, and `G` rows cost two allocations
//! rather than one `Vec` header *and* one heap allocation per row.
//!
//! # Why one type rather than one per caller
//!
//! [`AdjacencyGraph`](super::AdjacencyGraph) built this shape first for its
//! vertex→vertex, vertex→face and face→face relations, and the mesh repair
//! paths need the same shape for their component grouping. Two hand-rolled
//! copies of a counting-sort layout drift apart at the first fix (the exact
//! capacity invariant below is the kind that gets dropped in a copy), so the
//! storage lives here and both callers share it.
//!
//! # Construction contract
//!
//! [`PackedRows::from_counts`] takes the exact length of every row and returns
//! the storage together with one write cursor per row. [`PackedRows::write`]
//! then fills the buffer in row order. The two passes together are a counting
//! sort: `O(rows + items)` time, `O(rows + items)` space, no growth
//! reallocation, and no `Vec` header per row.
//!
//! # Invariant — exact row capacity
//!
//! `write` never grows a row. It asserts that the cursor for `row` is still
//! below `offsets[row + 1]`, so a caller that writes more items into a row than
//! it declared panics in debug builds instead of silently overwriting the next
//! row. A row's `counts` argument is a promise, and this is where it is
//! checked.

/// Contiguous rows addressed by an offset table.
///
/// The row count is `offsets.len() - 1`; row `r` occupies
/// `values[offsets[r]..offsets[r + 1]]`. The generic storage is independent of
/// mesh scalar precision and is instantiated only for the index types its
/// callers store.
pub(crate) struct PackedRows<T> {
    offsets: Box<[usize]>,
    values: Box<[T]>,
}

impl<T: Copy + Default> PackedRows<T> {
    /// Allocate one value buffer from exact row capacities and return cursors
    /// positioned at each row's start.
    pub(crate) fn from_counts(counts: Vec<usize>) -> (Self, Vec<usize>) {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);

        let mut total = 0usize;
        for count in counts {
            total = total
                .checked_add(count)
                .expect("invariant: packed-row storage size fits usize");
            offsets.push(total);
        }

        let cursors = offsets[..offsets.len() - 1].to_vec();
        let values = vec![T::default(); total].into_boxed_slice();

        (
            Self {
                offsets: offsets.into_boxed_slice(),
                values,
            },
            cursors,
        )
    }

    /// Write the next value in a row during the fill pass.
    #[inline]
    pub(crate) fn write(&mut self, cursors: &mut [usize], row: usize, value: T) {
        let cursor = cursors.get_mut(row).expect("invariant: packed row exists");
        let end = *self
            .offsets
            .get(row + 1)
            .expect("invariant: packed row end exists");
        let index = *cursor;
        debug_assert!(index < end, "invariant: packed row capacity is exact");
        *self
            .values
            .get_mut(index)
            .expect("invariant: packed row has remaining capacity") = value;
        *cursor = index + 1;
    }
}

impl<T> PackedRows<T> {
    /// Return a row, or an empty slice for an out-of-range ID.
    #[inline]
    pub(crate) fn get(&self, row: usize) -> &[T] {
        let Some(next) = row.checked_add(1) else {
            return &[];
        };

        match (self.offsets.get(row), self.offsets.get(next)) {
            (Some(&start), Some(&end)) => &self.values[start..end],
            _ => &[],
        }
    }

    /// Number of rows in the packed relation.
    #[inline]
    pub(crate) fn row_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The row offsets: `values[offsets()[r]..offsets()[r + 1]]` is row `r`.
    ///
    /// Exposed so a caller can index a *second* buffer laid out by the same
    /// partition (the component grouping keeps prepared face geometry beside
    /// its face indices) without duplicating the offset table.
    #[inline]
    pub(crate) fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The flat value buffer, in row order.
    #[inline]
    pub(crate) fn values(&self) -> &[T] {
        &self.values
    }
}

impl<T: Copy + Ord> PackedRows<T> {
    /// Sort and deduplicate every row, compacting the value buffer in place.
    pub(crate) fn sort_dedup(&mut self) {
        let row_count = self.row_count();
        let mut offsets = Vec::with_capacity(row_count + 1);
        offsets.push(0);
        let mut compacted = 0usize;

        for row in 0..row_count {
            let start = self.offsets[row];
            let end = self.offsets[row + 1];
            let unique_len = {
                let row_values = &mut self.values[start..end];
                row_values.sort_unstable();
                dedup_sorted(row_values)
            };

            if compacted != start {
                self.values
                    .copy_within(start..start + unique_len, compacted);
            }
            compacted += unique_len;
            offsets.push(compacted);
        }

        let values = std::mem::replace(&mut self.values, Vec::<T>::new().into_boxed_slice());
        let mut values = values.into_vec();
        values.truncate(compacted);
        self.values = values.into_boxed_slice();
        self.offsets = offsets.into_boxed_slice();
    }
}

/// Return the unique prefix length of a sorted slice while compacting it.
fn dedup_sorted<T: Copy + Eq>(values: &mut [T]) -> usize {
    let mut unique_len = 0usize;
    for read in 0..values.len() {
        if unique_len == 0 || values[read] != values[unique_len - 1] {
            if read != unique_len {
                values.swap(read, unique_len);
            }
            unique_len += 1;
        }
    }
    unique_len
}
