//! Column-major position bitboards and the per-piece geometry / kick translation tables.
//!
//! A position set for one rotation is `[u64; 10]`: word `c` is the box-left column, bit `r` is the
//! box-bottom row. "Box" = the bounding box of the piece's 4 cells, so for a piece at `(x, y)` in rotation
//! `rot`: `c = x + min_dx`, `r = y - max_dy` (both `>= 0` for every in-bounds position). Two rotations
//! with the same cells (I/S/Z/O symmetry) have the same box, so folding symmetric rotations is a plain OR.

// Column loops index several parallel arrays at once; iterator forms would obscure that.
#![allow(clippy::needless_range_loop)]

use std::sync::LazyLock;

use crate::board::{BOARD_HEIGHT, BOARD_WIDTH, Board};
use crate::kicks::{is_tst_or_fin_kick, kicks};
use crate::piece::{Piece, Rotation};

pub(crate) const W: usize = BOARD_WIDTH;
/// Valid rows of a column word.
pub(crate) const ROWS: u64 = (1u64 << BOARD_HEIGHT) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct Bits(pub [u64; W]);

impl Bits {
    pub const EMPTY: Bits = Bits([0; W]);

    #[inline(always)]
    pub fn any(&self) -> bool {
        self.0.iter().fold(0, |a, &w| a | w) != 0
    }

    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    #[inline(always)]
    pub fn and(&self, o: &Bits) -> Bits {
        let mut r = [0; W];
        for c in 0..W {
            r[c] = self.0[c] & o.0[c];
        }
        Bits(r)
    }

    #[inline(always)]
    pub fn or(&self, o: &Bits) -> Bits {
        let mut r = [0; W];
        for c in 0..W {
            r[c] = self.0[c] | o.0[c];
        }
        Bits(r)
    }

    #[inline(always)]
    pub fn andnot(&self, o: &Bits) -> Bits {
        let mut r = [0; W];
        for c in 0..W {
            r[c] = self.0[c] & !o.0[c];
        }
        Bits(r)
    }

    #[inline(always)]
    pub fn or_assign(&mut self, o: &Bits) {
        for c in 0..W {
            self.0[c] |= o.0[c];
        }
    }

    /// Translate every position by `(dc, dr)`: `out[c][r] = self[c - dc][r - dr]` (0 outside).
    #[inline(always)]
    pub fn shifted(&self, dc: i8, dr: i8) -> Bits {
        let mut r = [0; W];
        for (c, out) in r.iter_mut().enumerate() {
            let src = c as i32 - dc as i32;
            if (0..W as i32).contains(&src) {
                let v = self.0[src as usize];
                *out = if dr >= 0 { v << dr } else { v >> (-dr) };
            }
        }
        Bits(r)
    }

    #[inline(always)]
    pub fn set(&mut self, c: usize, r: usize) {
        self.0[c] |= 1 << r;
    }
}

/// Kogge–Stone occluded fill downward: every position reachable from `s` by moving down through `u`.
#[inline(always)]
pub(crate) fn fill_down(s: u64, u: u64) -> u64 {
    let mut f = s & u;
    let mut m = u;
    f |= m & (f >> 1);
    m &= m >> 1;
    f |= m & (f >> 2);
    m &= m >> 2;
    f |= m & (f >> 4);
    m &= m >> 4;
    f |= m & (f >> 8);
    m &= m >> 8;
    f |= m & (f >> 16);
    m &= m >> 16;
    f |= m & (f >> 32);
    f
}

/// Column bitboards of the board: bit `y` of word `x` = cell `(x, y)` filled.
#[inline]
pub(crate) fn columns(board: &Board) -> [u64; W] {
    let mut cols = [0u64; W];
    let top = board.top();
    let rows = board.rows();
    for (y, &row) in rows.iter().enumerate().take(top) {
        let mut m = row;
        while m != 0 {
            let x = m.trailing_zeros() as usize;
            cols[x] |= 1 << y;
            m &= m - 1;
        }
    }
    cols
}

/// Box geometry of one piece in one rotation.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Geo {
    /// `c = x + min_dx`.
    pub min_dx: i8,
    /// `r = y - max_dy`.
    pub max_dy: i8,
    /// Box width (columns): the box-left column ranges over `0..=10 - w`.
    pub w: u8,
    /// Cells relative to the box's bottom-left, `(column, row up)`.
    pub cells: [(u8, u8); 4],
}

/// One rotation direction from one rotation state: the target rotation and its tests (test 0 first)
/// as box translations, with a bit per test that is the T fin/TST kick.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RotTab {
    pub to: u8,
    pub n: u8,
    pub tr: [(i8, i8); 6],
    pub fin: u8,
}

pub(crate) struct Tables {
    pub geo: [[Geo; 4]; 7],
    /// `rot[piece][from][k]` for k = 0 CW, 1 CCW, 2 180 (the [`AMOUNTS`] order).
    pub rot: [[[RotTab; 3]; 4]; 7],
}

/// Rotation amounts in table order.
pub(crate) const AMOUNTS: [i8; 3] = [1, -1, 2];

pub(crate) static TABLES: LazyLock<Tables> = LazyLock::new(build_tables);

fn build_tables() -> Tables {
    let mut t = Tables { geo: [[Geo::default(); 4]; 7], rot: [[[RotTab::default(); 3]; 4]; 7] };
    for p in Piece::ALL {
        for r in Rotation::ALL {
            let sh = p.shape(r);
            let mut cells = [(0u8, 0u8); 4];
            for (i, &(dx, dy)) in sh.cells.iter().enumerate() {
                cells[i] = ((dx - sh.min_dx) as u8, (sh.max_dy - dy) as u8);
            }
            t.geo[p.index()][r.index()] =
                Geo { min_dx: sh.min_dx, max_dy: sh.max_dy, w: (sh.max_dx - sh.min_dx + 1) as u8, cells };
        }
        for a in Rotation::ALL {
            for (k, &amount) in AMOUNTS.iter().enumerate() {
                let b = a.add(amount);
                let ga = p.shape(a);
                let gb = p.shape(b);
                let mut rt = RotTab { to: b as u8, ..RotTab::default() };
                // Box translation for a test that moves the piece by (kx, -ky) in (x, y).
                let tr = |kx: i8, ky: i8| (kx + gb.min_dx - ga.min_dx, ga.max_dy - gb.max_dy - ky);
                rt.tr[0] = tr(0, 0);
                let list = kicks(p, a, b);
                for (i, &(kx, ky)) in list.iter().enumerate() {
                    rt.tr[i + 1] = tr(kx, ky);
                    if p == Piece::T && is_tst_or_fin_kick(a, b, Some((kx, ky))) {
                        rt.fin |= 1 << (i + 1);
                    }
                }
                rt.n = list.len() as u8 + 1;
                t.rot[p.index()][a.index()][k] = rt;
            }
        }
    }
    t
}

/// Usable map of one rotation: positions where the piece fits, from the empty-cell columns `empty`.
#[inline(always)]
pub(crate) fn usable(empty: &[u64; W], g: &Geo) -> Bits {
    let mut u = [0u64; W];
    let last = W - g.w as usize;
    for (c, out) in u.iter_mut().enumerate().take(last + 1) {
        let mut m = ROWS;
        for &(bx, by) in &g.cells {
            m &= empty[c + bx as usize] >> by;
        }
        *out = m;
    }
    Bits(u)
}
