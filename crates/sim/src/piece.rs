//! Tetromino identities, rotation states and shapes (RULES.md §3.1, §3.2).
//!
//! Shapes are copied from triangle.js `src/engine/utils/tetromino/data.ts` (SRS matrices). Each cell is
//! `(dx, dy)` with **dy pointing down**, inside the piece's `w×w` matrix. A piece at location `(x, y)`
//! (`y` = `floor(loc.y)`, the board row of matrix row 0) occupies board cells `(x + dx, y - dy)`.

use crate::board::{BOARD_HEIGHT, BOARD_WIDTH};

/// A tetromino. The discriminant order is arbitrary; the bag order is [`Piece::BAG_ORDER`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum Piece {
    I = 0,
    O = 1,
    T = 2,
    L = 3,
    J = 4,
    S = 5,
    Z = 6,
}

impl Piece {
    pub const ALL: [Piece; 7] = [Piece::I, Piece::O, Piece::T, Piece::L, Piece::J, Piece::S, Piece::Z];

    /// The array the 7-bag shuffles (`src/engine/queue/rng/bag7.ts`): `[Z, L, O, S, I, J, T]`.
    pub const BAG_ORDER: [Piece; 7] = [Piece::Z, Piece::L, Piece::O, Piece::S, Piece::I, Piece::J, Piece::T];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Lowercase letter, as triangle's `Mino` values (`"i"`, `"o"`, ...).
    pub const fn to_char(self) -> char {
        match self {
            Piece::I => 'i',
            Piece::O => 'o',
            Piece::T => 't',
            Piece::L => 'l',
            Piece::J => 'j',
            Piece::S => 's',
            Piece::Z => 'z',
        }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        Some(match c.to_ascii_lowercase() {
            'i' => Piece::I,
            'o' => Piece::O,
            't' => Piece::T,
            'l' => Piece::L,
            'j' => Piece::J,
            's' => Piece::S,
            'z' => Piece::Z,
            _ => return None,
        })
    }

    /// Spawn location `[floor(10/2 - w/2), floor(20 + 2.04)]` (RULES.md §3.2).
    pub const fn spawn_x(self) -> i8 {
        match self {
            Piece::O => 4,
            _ => 3,
        }
    }

    #[inline]
    pub fn shape(self, rot: Rotation) -> &'static Shape {
        &SHAPES[self.index()][rot.index()]
    }
}

/// Row of matrix row 0 at spawn: `floor(boardHeight + 2.04)` = 22.
pub const SPAWN_Y: i8 = 22;

/// Rotation state. `North` = spawn (0), `East` = CW (1, "R"), `South` = 180 (2), `West` = CCW (3, "L").
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Rotation {
    #[default]
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

impl Rotation {
    pub const ALL: [Rotation; 4] = [Rotation::North, Rotation::East, Rotation::South, Rotation::West];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_u8(r: u8) -> Rotation {
        Rotation::ALL[(r & 3) as usize]
    }

    /// `rotation + amount` mod 4 (amount may be negative).
    #[inline]
    pub const fn add(self, amount: i8) -> Rotation {
        Rotation::from_u8((self as i8 + amount).rem_euclid(4) as u8)
    }

    #[inline]
    pub const fn cw(self) -> Rotation {
        self.add(1)
    }

    #[inline]
    pub const fn ccw(self) -> Rotation {
        self.add(-1)
    }

    #[inline]
    pub const fn flip(self) -> Rotation {
        self.add(2)
    }
}

/// Precomputed collision data for one piece in one rotation.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Cells `(dx, dy)` with dy down, in the engine's order.
    pub cells: [(i8, i8); 4],
    pub min_dx: i8,
    pub max_dx: i8,
    pub min_dy: i8,
    pub max_dy: i8,
    /// Number of occupied matrix rows (`max_dy - min_dy + 1`).
    pub height: u8,
    /// `rows[k]` = bitmask of cells with `dy == min_dy + k`; bit `dx - min_dx`.
    pub rows: [u16; 4],
}

impl Shape {
    const fn new(cells: [(i8, i8); 4]) -> Shape {
        let mut min_dx = i8::MAX;
        let mut max_dx = i8::MIN;
        let mut min_dy = i8::MAX;
        let mut max_dy = i8::MIN;
        let mut i = 0;
        while i < 4 {
            let (dx, dy) = cells[i];
            if dx < min_dx {
                min_dx = dx;
            }
            if dx > max_dx {
                max_dx = dx;
            }
            if dy < min_dy {
                min_dy = dy;
            }
            if dy > max_dy {
                max_dy = dy;
            }
            i += 1;
        }
        let mut rows = [0u16; 4];
        let mut i = 0;
        while i < 4 {
            let (dx, dy) = cells[i];
            rows[(dy - min_dy) as usize] |= 1 << (dx - min_dx);
            i += 1;
        }
        Shape { cells, min_dx, max_dx, min_dy, max_dy, height: (max_dy - min_dy + 1) as u8, rows }
    }

    /// Absolute board cells `(x + dx, y - dy)`.
    #[inline]
    pub fn cells_at(&self, x: i8, y: i8) -> [(i8, i8); 4] {
        let c = &self.cells;
        [(x + c[0].0, y - c[0].1), (x + c[1].0, y - c[1].1), (x + c[2].0, y - c[2].1), (x + c[3].0, y - c[3].1)]
    }

    /// True if `(x, y)` keeps every cell inside the 10×40 field.
    #[inline]
    pub fn in_bounds(&self, x: i8, y: i8) -> bool {
        x + self.min_dx >= 0
            && x + self.max_dx < BOARD_WIDTH as i8
            && y - self.max_dy >= 0
            && y - self.min_dy < BOARD_HEIGHT as i8
    }
}

const fn s(cells: [(i8, i8); 4]) -> Shape {
    Shape::new(cells)
}

/// `SHAPES[piece][rotation]`, from `tetromino/data.ts` (I `:365`, O `:277`, T `:453`, L `:233`, J `:409`,
/// S `:321`, Z `:189`).
pub static SHAPES: [[Shape; 4]; 7] = [
    // I
    [
        s([(0, 1), (1, 1), (2, 1), (3, 1)]),
        s([(2, 0), (2, 1), (2, 2), (2, 3)]),
        s([(3, 2), (2, 2), (1, 2), (0, 2)]),
        s([(1, 3), (1, 2), (1, 1), (1, 0)]),
    ],
    // O
    [
        s([(0, 0), (1, 0), (0, 1), (1, 1)]),
        s([(1, 0), (1, 1), (0, 0), (0, 1)]),
        s([(1, 1), (0, 1), (1, 0), (0, 0)]),
        s([(0, 1), (0, 0), (1, 1), (1, 0)]),
    ],
    // T
    [
        s([(1, 0), (0, 1), (1, 1), (2, 1)]),
        s([(2, 1), (1, 0), (1, 1), (1, 2)]),
        s([(1, 2), (2, 1), (1, 1), (0, 1)]),
        s([(0, 1), (1, 2), (1, 1), (1, 0)]),
    ],
    // L
    [
        s([(2, 0), (0, 1), (1, 1), (2, 1)]),
        s([(2, 2), (1, 0), (1, 1), (1, 2)]),
        s([(0, 2), (2, 1), (1, 1), (0, 1)]),
        s([(0, 0), (1, 2), (1, 1), (1, 0)]),
    ],
    // J
    [
        s([(0, 0), (0, 1), (1, 1), (2, 1)]),
        s([(2, 0), (1, 0), (1, 1), (1, 2)]),
        s([(2, 2), (2, 1), (1, 1), (0, 1)]),
        s([(0, 2), (1, 2), (1, 1), (1, 0)]),
    ],
    // S
    [
        s([(1, 0), (2, 0), (0, 1), (1, 1)]),
        s([(2, 1), (2, 2), (1, 0), (1, 1)]),
        s([(1, 2), (0, 2), (2, 1), (1, 1)]),
        s([(0, 1), (0, 0), (1, 2), (1, 1)]),
    ],
    // Z
    [
        s([(0, 0), (1, 0), (1, 1), (2, 1)]),
        s([(2, 0), (2, 1), (1, 1), (1, 2)]),
        s([(2, 2), (1, 2), (1, 1), (0, 1)]),
        s([(0, 2), (0, 1), (1, 1), (1, 0)]),
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<(i8, i8)>) -> Vec<(i8, i8)> {
        v.sort_by_key(|&(x, y)| (y, x));
        v
    }

    #[test]
    fn spawn_cells_match_rules_table() {
        // RULES.md §3.2 "Absolute spawn cells".
        let expect: [(Piece, [(i8, i8); 4]); 7] = [
            (Piece::I, [(3, 21), (4, 21), (5, 21), (6, 21)]),
            (Piece::J, [(3, 22), (3, 21), (4, 21), (5, 21)]),
            (Piece::L, [(5, 22), (3, 21), (4, 21), (5, 21)]),
            (Piece::O, [(4, 22), (5, 22), (4, 21), (5, 21)]),
            (Piece::S, [(4, 22), (5, 22), (3, 21), (4, 21)]),
            (Piece::T, [(4, 22), (3, 21), (4, 21), (5, 21)]),
            (Piece::Z, [(3, 22), (4, 22), (4, 21), (5, 21)]),
        ];
        for (p, cells) in expect {
            let got = p.shape(Rotation::North).cells_at(p.spawn_x(), SPAWN_Y);
            assert_eq!(sorted(got.to_vec()), sorted(cells.to_vec()), "{p:?}");
        }
    }

    #[test]
    fn rows_masks_consistent() {
        for p in Piece::ALL {
            for r in Rotation::ALL {
                let sh = p.shape(r);
                let n: u32 = sh.rows.iter().map(|m| m.count_ones()).sum();
                assert_eq!(n, 4);
                for &(dx, dy) in &sh.cells {
                    assert!(sh.rows[(dy - sh.min_dy) as usize] & (1 << (dx - sh.min_dx)) != 0);
                }
            }
        }
    }

    #[test]
    fn rotation_arith() {
        assert_eq!(Rotation::North.ccw(), Rotation::West);
        assert_eq!(Rotation::West.cw(), Rotation::North);
        assert_eq!(Rotation::East.flip(), Rotation::West);
    }
}
