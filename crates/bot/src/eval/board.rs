//! Board features for the hand-written evaluation. All bit-parallel over the `u16` rows; one pass up and
//! one pass down the stack, no allocation.

use tetris_sim::Board;
use tetris_sim::board::{BOARD_WIDTH, FULL_ROW};

/// Shape and setup features of one board (natural units: cells, rows, lines).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoardFeatures {
    /// Empty cells with a filled cell above them in the same column.
    pub holes: u32,
    /// Rows containing at least one hole.
    pub hole_rows: u32,
    /// Filled cells with a hole below them in the same column.
    pub covered: u32,
    /// Neighbour height differences, the well column excluded.
    pub bump: u32,
    pub bump_sq: u32,
    pub max_height: u32,
    /// Sum of column heights (mean = `sum_height / 10`).
    pub sum_height: u32,
    pub row_trans: u32,
    pub col_trans: u32,
    /// Deepest well: column and depth (`min(neighbour heights) - height`, walls count as infinitely high).
    pub well_col: u8,
    pub well_depth: u32,
    /// Rows from the well bottom up that are full except the well column (capped at 4).
    pub quad_ready: u32,
    /// Best T-spin double slot: 0 if none, else 0.5 + the number of lines a T-spin would clear (1..2).
    pub tslot: f32,
    /// No holes, stack ≤ 4 rows, even cell count.
    pub pc_ready: bool,
}

const WALLS: u32 = 1 | 1 << (BOARD_WIDTH + 1);

pub fn board_features(b: &Board) -> BoardFeatures {
    let rows = b.rows();
    let h = b.heights();
    let top = b.top();
    let mut f = BoardFeatures::default();

    let mut above = 0u16;
    for y in (0..top).rev() {
        let r = rows[y];
        let holes = above & !r & FULL_ROW;
        f.holes += holes.count_ones();
        f.hole_rows += (holes != 0) as u32;
        above |= r;
    }
    let mut empty_below = 0u16;
    let mut prev = FULL_ROW; // the floor
    for &r in &rows[..top] {
        f.covered += (r & empty_below).count_ones();
        empty_below |= !r & FULL_ROW;
        let w = WALLS | (r as u32) << 1;
        f.row_trans += ((w ^ (w >> 1)) & 0x7ff).count_ones();
        f.col_trans += (r ^ prev).count_ones();
        prev = r;
    }
    f.col_trans += prev.count_ones(); // the surface (row `top` is empty)

    let mut best_depth = -1i32;
    for x in 0..BOARD_WIDTH {
        let hx = h[x] as i32;
        f.sum_height += hx as u32;
        f.max_height = f.max_height.max(hx as u32);
        let l = if x == 0 { 99 } else { h[x - 1] as i32 };
        let r = if x == BOARD_WIDTH - 1 { 99 } else { h[x + 1] as i32 };
        let d = l.min(r) - hx;
        if d > best_depth {
            best_depth = d;
            f.well_col = x as u8;
        }
    }
    f.well_depth = best_depth.max(0) as u32;
    let wc = f.well_col as usize;
    for x in 0..BOARD_WIDTH - 1 {
        if x == wc || x + 1 == wc {
            continue;
        }
        let d = (h[x] as i32 - h[x + 1] as i32).unsigned_abs();
        f.bump += d;
        f.bump_sq += d * d;
    }
    let wbit = 1u16 << wc;
    let mut y = h[wc] as usize;
    while y < top && f.quad_ready < 4 && rows[y] | wbit == FULL_ROW {
        f.quad_ready += 1;
        y += 1;
    }

    f.tslot = best_tsd_slot(b);
    f.pc_ready = f.holes == 0 && top <= 4 && rows[..top].iter().map(|r| r.count_ones()).sum::<u32>() % 2 == 0;
    f
}

/// Best T-spin double slot (T pointing down into the slot, one overhang corner on top).
///
/// For a T occupying columns `x..x+3` with its stem at `(x+1, y)`: the stem cell and the three cells of
/// row `y+1` are empty, both bottom corners `(x, y)`, `(x+2, y)` are filled (front corners → a full
/// T-spin), exactly one top corner `(x, y+2)` / `(x+2, y+2)` is filled (the overhang that makes it a
/// spin), the cell above the stem and the open side are free so the T can come in. The value is 0.5 for
/// the shape plus one per line (row `y`, row `y+1`) that the spin would clear right now.
pub fn best_tsd_slot(b: &Board) -> f32 {
    let rows = b.rows();
    let h = b.heights();
    let mut best = 0.0f32;
    for x in 0..BOARD_WIDTH - 2 {
        let y = h[x + 1] as usize; // stem row: the centre column's surface
        if y + 3 >= rows.len() {
            continue;
        }
        let (r0, r1, r2) = (rows[y], rows[y + 1], rows[y + 2]);
        let bit = |r: u16, c: usize| r >> c & 1 != 0;
        if !bit(r0, x) || !bit(r0, x + 2) {
            continue;
        }
        if r1 & (0b111 << x) != 0 || bit(r2, x + 1) {
            continue;
        }
        let (tl, tr) = (bit(r2, x), bit(r2, x + 2));
        if tl == tr {
            continue;
        }
        // Entry: the open side must be clear above row y+1.
        let open = if tl { x + 2 } else { x };
        if h[open] as usize > y + 1 {
            continue;
        }
        let mut v = 0.5;
        if r0 | 1 << (x + 1) == FULL_ROW {
            v += 1.0;
        }
        if r1 | 0b111 << x == FULL_ROW {
            v += 1.0;
        }
        best = best.max(v);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::movegen::{self, Mode};
    use tetris_sim::{Falling, Piece, Spin};

    #[test]
    fn empty_board() {
        let f = board_features(&Board::new());
        assert_eq!((f.holes, f.hole_rows, f.covered, f.bump, f.max_height), (0, 0, 0, 0, 0));
        assert_eq!(f.col_trans, 10, "floor-to-empty in every column");
        assert!(f.pc_ready);
        assert_eq!(f.tslot, 0.0);
    }

    #[test]
    fn holes_covered_transitions() {
        let b = Board::from_ascii(
            "
            #.........
            #.##......
            ..##......
            ###.######
            ",
        );
        let f = board_features(&b);
        // Column 0: hole at row 1 (covered by rows 2, 3). Column 3: hole at row 0 under rows 1, 2.
        assert_eq!(f.holes, 2);
        assert_eq!(f.hole_rows, 2);
        assert_eq!(f.covered, 2 + 2);
        assert_eq!(f.max_height, 4);
        assert_eq!(f.sum_height, 4 + 1 + 3 + 3 + 6);
        // Row 0 "###.######" with walls: 2 transitions. Row 1 "..##......": wall|..|##|......|wall = 4.
        // Row 2 "#.##......": 1|#.##......|1 -> #→. . →# #→. .→wall = 4. Row 3 "#.........": 2.
        assert_eq!(f.row_trans, 2 + 4 + 4 + 2);
        assert!(!f.pc_ready);
    }

    #[test]
    fn well_and_quad_ready() {
        let b = Board::from_ascii(
            "
            #########.
            #########.
            #########.
            ",
        );
        let f = board_features(&b);
        assert_eq!((f.well_col, f.well_depth, f.quad_ready), (9, 3, 3));
        assert_eq!(f.bump, 0, "the well column is excluded");
        assert_eq!(f.holes, 0);
    }

    #[test]
    fn tsd_slot_detected_and_movegen_agrees() {
        // Classic TSD: overhang at (2, 2), slot at column 3.
        let b = Board::from_ascii(
            "
            ..#.......
            ...._.....
            ###.######
            ",
        );
        // The same slot with row 1 full except the T's three cells: a ready double.
        let ready = Board::from_ascii(
            "
            ..#.......
            ##...#####
            ###.######
            ",
        );
        assert_eq!(best_tsd_slot(&b), 1.5, "shape + one ready line (row 0)");
        assert_eq!(best_tsd_slot(&ready), 2.5);
        // Movegen finds a T-spin double there.
        let mut buf = Vec::new();
        movegen::generate_from(&ready, Falling::spawn(Piece::T), Spin::None, Mode::Sonic, false, &mut buf);
        let mut found = false;
        for p in &buf {
            let mut s = tetris_sim::PlayerState::tl(1);
            s.board = ready;
            s.falling = s.spawn_falling(Piece::T).0;
            let out = s.apply_placement(*p);
            found |= out.spin == Spin::Normal && out.lines == 2;
        }
        assert!(found);
        // No overhang: no slot. Both overhangs: not counted (needs a TST-style kick).
        let flat = Board::from_ascii("###.######");
        assert_eq!(best_tsd_slot(&flat), 0.0);
        let closed = Board::from_ascii(
            "
            ..#.#.....
            ##...#####
            ###.######
            ",
        );
        assert_eq!(best_tsd_slot(&closed), 0.0);
    }
}
