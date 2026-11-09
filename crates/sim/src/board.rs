//! The playfield: 10 columns × 40 rows (20 visible + 20 buffer), row 0 at the bottom (RULES.md §2).
//!
//! Each row is a `u16` with bit `x` = column `x`. Column heights are cached and updated incrementally.
//! Garbage rows are tracked with one bit per row (`garbage_rows`): rows only ever move as whole units
//! (line clears and garbage insertion), so "this row contains garbage cells" is a per-row property.

use crate::piece::Shape;

pub const BOARD_WIDTH: usize = 10;
pub const BOARD_HEIGHT: usize = 40;
pub const VISIBLE_HEIGHT: usize = 20;
pub const FULL_ROW: u16 = (1 << BOARD_WIDTH) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    rows: [u16; BOARD_HEIGHT],
    /// `heights[x]` = 1 + index of the highest filled cell in column `x` (0 if empty).
    heights: [u8; BOARD_WIDTH],
    /// Bit `y` set ⇔ row `y` contains garbage cells.
    garbage_rows: u64,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Board {{")?;
        for y in (0..self.top()).rev() {
            let r = self.rows[y];
            let s: String = (0..BOARD_WIDTH)
                .map(|x| {
                    if r >> x & 1 == 0 {
                        '.'
                    } else if self.garbage_rows >> y & 1 == 1 {
                        'G'
                    } else {
                        '#'
                    }
                })
                .collect();
            writeln!(f, "  {y:2} |{s}|")?;
        }
        write!(f, "}}")
    }
}

/// Result of [`Board::clear_lines`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Cleared {
    pub lines: u8,
    /// Cleared rows that contained garbage (`garbageCleared`).
    pub garbage: u8,
    /// Bit `y` set ⇔ row `y` (pre-clear index) was cleared.
    pub mask: u64,
}

impl Board {
    pub const fn new() -> Board {
        Board { rows: [0; BOARD_HEIGHT], heights: [0; BOARD_WIDTH], garbage_rows: 0 }
    }

    /// Build from rows (bottom first) and a garbage-row mask. Heights are computed.
    pub fn from_rows(rows: &[u16], garbage_rows: u64) -> Board {
        let mut b = Board::new();
        for (y, &r) in rows.iter().enumerate().take(BOARD_HEIGHT) {
            b.rows[y] = r & FULL_ROW;
        }
        b.garbage_rows = garbage_rows & ((1u64 << BOARD_HEIGHT) - 1);
        b.recompute_heights();
        b
    }

    /// Parse rows written top-to-bottom with `.`/`_`/space = empty, anything else = filled
    /// (`G` marks a garbage row). Used for fixtures and tests; not a hot-path constructor.
    pub fn from_ascii(text: &str) -> Board {
        let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let mut rows = Vec::new();
        let mut garbage = 0u64;
        for (y, line) in lines.iter().rev().enumerate() {
            let mut r = 0u16;
            for (x, c) in line.chars().take(BOARD_WIDTH).enumerate() {
                if !matches!(c, '.' | '_' | ' ') {
                    r |= 1 << x;
                }
                if c == 'G' {
                    garbage |= 1 << y;
                }
            }
            rows.push(r);
        }
        Board::from_rows(&rows, garbage)
    }

    #[inline]
    pub fn rows(&self) -> &[u16; BOARD_HEIGHT] {
        &self.rows
    }

    #[inline]
    pub fn row(&self, y: usize) -> u16 {
        self.rows[y]
    }

    #[inline]
    pub fn heights(&self) -> &[u8; BOARD_WIDTH] {
        &self.heights
    }

    #[inline]
    pub fn height(&self, x: usize) -> u8 {
        self.heights[x]
    }

    #[inline]
    pub fn garbage_rows(&self) -> u64 {
        self.garbage_rows
    }

    /// Height of the stack: 1 + highest non-empty row (0 if empty).
    #[inline]
    pub fn top(&self) -> usize {
        *self.heights.iter().max().unwrap() as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heights == [0; BOARD_WIDTH]
    }

    pub fn cell_count(&self) -> u32 {
        self.rows.iter().map(|r| r.count_ones()).sum()
    }

    /// Filled cell (in bounds only).
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.rows[y] >> x & 1 != 0
    }

    /// `board.occupied(x, y)`: true out of bounds (walls, floor, ceiling) or filled.
    #[inline]
    pub fn occupied(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x >= BOARD_WIDTH as i32 || y >= BOARD_HEIGHT as i32 {
            return true;
        }
        self.rows[y as usize] >> x & 1 != 0
    }

    /// `legal(cells)` for a piece shape at `(x, y)`: all cells in bounds and empty.
    #[inline]
    pub fn fits(&self, shape: &Shape, x: i8, y: i8) -> bool {
        if !shape.in_bounds(x, y) {
            return false;
        }
        let left = (x + shape.min_dx) as u32;
        let top = (y - shape.min_dy) as usize;
        let mut k = 0;
        while k < shape.height as usize {
            if self.rows[top - k] & (shape.rows[k] << left) != 0 {
                return false;
            }
            k += 1;
        }
        true
    }

    /// Lowest `y' <= y` reachable by moving straight down from `(x, y)` (sonic drop). Assumes nothing
    /// about the start position (like the engine, it only tests `y'-1` each step).
    #[inline]
    pub fn drop_y(&self, shape: &Shape, x: i8, y: i8) -> i8 {
        // Fast path: if the piece is above the stack in all its columns, the landing row follows from
        // the column heights.
        if shape.in_bounds(x, y) {
            let mut land = i8::MIN;
            let mut above = true;
            for &(dx, dy) in &shape.cells {
                let col = (x + dx) as usize;
                let h = self.heights[col] as i8;
                if y - dy < h {
                    above = false;
                    break;
                }
                // Cell row y' - dy must be >= h  =>  y' >= h + dy.
                land = land.max(h + dy);
            }
            if above {
                return land;
            }
        }
        let mut y = y;
        while self.fits(shape, x, y - 1) {
            y -= 1;
        }
        y
    }

    /// Set cells (out-of-bounds cells are skipped, like `Board.add`).
    #[inline]
    pub fn place(&mut self, cells: &[(i8, i8); 4]) {
        for &(x, y) in cells {
            if x < 0 || y < 0 || x as usize >= BOARD_WIDTH || y as usize >= BOARD_HEIGHT {
                continue;
            }
            self.rows[y as usize] |= 1 << x;
            let h = &mut self.heights[x as usize];
            if *h < y as u8 + 1 {
                *h = y as u8 + 1;
            }
        }
    }

    /// Clear full rows among `[lo, hi]` (inclusive, clamped). Only rows touched by the last piece can be
    /// full, so callers pass the piece's row range; [`Board::clear_all_lines`] scans everything.
    #[inline]
    pub fn clear_lines_in(&mut self, lo: i8, hi: i8) -> Cleared {
        let lo = lo.max(0) as usize;
        let hi = (hi.max(0) as usize).min(BOARD_HEIGHT - 1);
        let mut mask = 0u64;
        for y in lo..=hi {
            if self.rows[y] == FULL_ROW {
                mask |= 1 << y;
            }
        }
        if mask == 0 {
            return Cleared::default();
        }
        self.remove_rows(mask)
    }

    /// Clear every full row on the board.
    pub fn clear_all_lines(&mut self) -> Cleared {
        self.clear_lines_in(0, BOARD_HEIGHT as i8 - 1)
    }

    fn remove_rows(&mut self, mask: u64) -> Cleared {
        let lines = mask.count_ones() as u8;
        let garbage = (mask & self.garbage_rows).count_ones() as u8;
        let lo = mask.trailing_zeros() as usize;
        let mut w = lo;
        let mut g = self.garbage_rows & ((1u64 << lo) - 1);
        for r in lo..BOARD_HEIGHT {
            if mask >> r & 1 != 0 {
                continue;
            }
            self.rows[w] = self.rows[r];
            g |= (self.garbage_rows >> r & 1) << w;
            w += 1;
        }
        for r in w..BOARD_HEIGHT {
            self.rows[r] = 0;
        }
        self.garbage_rows = g;
        self.recompute_heights();
        Cleared { lines, garbage, mask }
    }

    /// `Board.insertGarbage` for one line (amount 1, hole size 1): insert a garbage row with a hole at
    /// `column` at the bottom, then delete old row 37 (the engine's `splice(fullHeight - amount - 1,
    /// amount)`), so old rows 38 and 39 stay in place. RULES.md §2.
    #[inline]
    pub fn insert_garbage(&mut self, column: u8) {
        let row = FULL_ROW & !(1u16 << column);
        let high = self.rows[37] | self.rows[38] | self.rows[39];
        let r38 = self.rows[38];
        let r39 = self.rows[39];
        self.rows.copy_within(0..37, 1);
        self.rows[0] = row;
        self.rows[38] = r38;
        self.rows[39] = r39;
        let keep_hi = self.garbage_rows & (0b11 << 38);
        self.garbage_rows = ((self.garbage_rows << 1) & ((1u64 << 38) - 1)) | 1 | keep_hi;
        if high != 0 {
            self.recompute_heights();
        } else {
            for x in 0..BOARD_WIDTH {
                let h = &mut self.heights[x];
                if *h > 0 {
                    *h += 1;
                } else if x != column as usize {
                    *h = 1;
                }
            }
        }
    }

    fn recompute_heights(&mut self) {
        let mut remaining = FULL_ROW;
        let mut h = [0u8; BOARD_WIDTH];
        for y in (0..BOARD_HEIGHT).rev() {
            let m = self.rows[y] & remaining;
            if m != 0 {
                let mut bits = m;
                while bits != 0 {
                    let x = bits.trailing_zeros() as usize;
                    h[x] = y as u8 + 1;
                    bits &= bits - 1;
                }
                remaining &= !m;
                if remaining == 0 {
                    break;
                }
            }
        }
        self.heights = h;
    }

    /// Debug invariant: cached heights agree with the rows.
    pub fn check_invariants(&self) -> bool {
        let mut b = *self;
        b.recompute_heights();
        b.heights == self.heights && self.rows.iter().all(|r| r & !FULL_ROW == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Piece, Rotation};

    #[test]
    fn clear_and_heights() {
        let mut b = Board::from_ascii(
            "
            #.........
            ##########
            GGGG.GGGGG
            ##########
            ",
        );
        assert_eq!(b.heights()[0], 4);
        let c = b.clear_all_lines();
        assert_eq!(c.lines, 2);
        assert_eq!(c.garbage, 0);
        assert_eq!(c.mask, 0b101);
        assert_eq!(b.row(0), FULL_ROW & !(1 << 4));
        assert_eq!(b.row(1), 1);
        assert_eq!(b.garbage_rows(), 1);
        assert_eq!(b.heights()[0], 2);
        assert_eq!(b.heights()[4], 0);
        assert!(b.check_invariants());
    }

    #[test]
    fn garbage_insert_quirk_keeps_rows_38_39() {
        let mut rows = [0u16; 40];
        rows[36] = 1;
        rows[37] = 2;
        rows[38] = 4;
        rows[39] = 8;
        let mut b = Board::from_rows(&rows, 0);
        b.insert_garbage(3);
        assert_eq!(b.row(0), FULL_ROW & !(1 << 3));
        assert_eq!(b.row(37), 1); // old 36
        assert_eq!(b.row(38), 4); // old 38 stays
        assert_eq!(b.row(39), 8); // old 39 stays; old 37 deleted
        assert!(b.check_invariants());
        assert_eq!(b.garbage_rows(), 1);
    }

    #[test]
    fn garbage_insert_heights_incremental() {
        let mut b = Board::from_ascii("..##......");
        b.insert_garbage(2);
        assert_eq!(b.heights(), &[1, 1, 2, 2, 1, 1, 1, 1, 1, 1]);
        b.insert_garbage(0);
        assert_eq!(b.heights(), &[2, 2, 3, 3, 2, 2, 2, 2, 2, 2]);
        assert!(b.check_invariants());
    }

    #[test]
    fn drop_fast_and_slow_agree() {
        let b = Board::from_ascii(
            "
            ...###....
            ..........
            #.........
            ",
        );
        for p in Piece::ALL {
            for r in Rotation::ALL {
                let sh = p.shape(r);
                for x in -2..10 {
                    if !b.fits(sh, x, 30) {
                        continue;
                    }
                    let mut y = 30;
                    while b.fits(sh, x, y - 1) {
                        y -= 1;
                    }
                    assert_eq!(b.drop_y(sh, x, 30), y, "{p:?} {r:?} x={x}");
                }
            }
        }
    }
}
