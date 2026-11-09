//! Piece-colour board for trace output. It mirrors [`crate::Board`] through the [`LockObserver`] hooks,
//! so the hot path keeps only bitboards.

use crate::board::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::piece::Piece;
use crate::state::LockObserver;

/// `cells[y][x]`: `b'.'` empty, `b'I'..b'Z'` piece, `b'G'` garbage.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ColorBoard {
    pub cells: [[u8; BOARD_WIDTH]; BOARD_HEIGHT],
    /// `(column, cid)` of each garbage line inserted since the last [`ColorBoard::take_added`].
    pub added: Vec<(u8, u32)>,
}

impl Default for ColorBoard {
    fn default() -> Self {
        ColorBoard { cells: [[b'.'; BOARD_WIDTH]; BOARD_HEIGHT], added: Vec::new() }
    }
}

impl ColorBoard {
    pub fn take_added(&mut self) -> Vec<(u8, u32)> {
        std::mem::take(&mut self.added)
    }

    /// Rows bottom → top up to the highest non-empty row.
    pub fn rows(&self) -> Vec<String> {
        let top = (0..BOARD_HEIGHT).rev().find(|&y| self.cells[y].iter().any(|&c| c != b'.')).map_or(0, |y| y + 1);
        (0..top).map(|y| String::from_utf8(self.cells[y].to_vec()).unwrap()).collect()
    }

    /// Bitmask rows, for cross-checking against the bitboard.
    pub fn bits(&self) -> [u16; BOARD_HEIGHT] {
        let mut out = [0u16; BOARD_HEIGHT];
        for (y, row) in self.cells.iter().enumerate() {
            for (x, &c) in row.iter().enumerate() {
                if c != b'.' {
                    out[y] |= 1 << x;
                }
            }
        }
        out
    }
}

impl LockObserver for ColorBoard {
    fn placed(&mut self, piece: Piece, cells: &[(i8, i8); 4]) {
        let c = piece.to_char().to_ascii_uppercase() as u8;
        for &(x, y) in cells {
            if x >= 0 && y >= 0 && (x as usize) < BOARD_WIDTH && (y as usize) < BOARD_HEIGHT {
                self.cells[y as usize][x as usize] = c;
            }
        }
    }

    fn cleared(&mut self, mask: u64) {
        let mut w = 0;
        for r in 0..BOARD_HEIGHT {
            if mask >> r & 1 != 0 {
                continue;
            }
            self.cells[w] = self.cells[r];
            w += 1;
        }
        for r in w..BOARD_HEIGHT {
            self.cells[r] = [b'.'; BOARD_WIDTH];
        }
    }

    fn garbage_line(&mut self, column: u8, cid: u32) {
        // Same row-37 quirk as `Board::insert_garbage`.
        let r38 = self.cells[38];
        let r39 = self.cells[39];
        self.cells.copy_within(0..37, 1);
        let mut row = [b'G'; BOARD_WIDTH];
        row[column as usize] = b'.';
        self.cells[0] = row;
        self.cells[38] = r38;
        self.cells[39] = r39;
        self.added.push((column, cid));
    }
}
