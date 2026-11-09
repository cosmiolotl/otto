//! SRS+ kick tables, literal from triangle.js `src/engine/utils/kicks/data.ts:430-562` (RULES.md §4).
//!
//! Every entry is `(dx, dy)` with **dy = +1 meaning DOWN one row**, exactly as in the source. Test 0 (no
//! offset) is implicit and not listed. A successful test `(kx, ky)` moves the piece to
//! `(x + kx, y - ky)` (`y` = `floor(loc.y)`; see `performKick`).

use crate::piece::{Piece, Rotation};

type K = &'static [(i8, i8)];

const EMPTY: K = &[];

// J, L, S, T, Z, O: `kicks` (`data.ts:431-508`).
const K01: K = &[(-1, 0), (-1, -1), (0, 2), (-1, 2)];
const K10: K = &[(1, 0), (1, 1), (0, -2), (1, -2)];
const K12: K = &[(1, 0), (1, 1), (0, -2), (1, -2)];
const K21: K = &[(-1, 0), (-1, -1), (0, 2), (-1, 2)];
const K23: K = &[(1, 0), (1, -1), (0, 2), (1, 2)];
const K32: K = &[(-1, 0), (-1, 1), (0, -2), (-1, -2)];
const K30: K = &[(-1, 0), (-1, 1), (0, -2), (-1, -2)];
const K03: K = &[(1, 0), (1, -1), (0, 2), (1, 2)];
const K02: K = &[(0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0)];
const K13: K = &[(1, 0), (1, -2), (1, -1), (0, -2), (0, -1)];
const K20: K = &[(0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)];
const K31: K = &[(-1, 0), (-1, -2), (-1, -1), (0, -2), (0, -1)];

// I: `i_kicks` (`data.ts:509-562`).
const I01: K = &[(1, 0), (-2, 0), (-2, 1), (1, -2)];
const I10: K = &[(-1, 0), (2, 0), (-1, 2), (2, -1)];
const I12: K = &[(-1, 0), (2, 0), (-1, -2), (2, 1)];
const I21: K = &[(-2, 0), (1, 0), (-2, -1), (1, 2)];
const I23: K = &[(2, 0), (-1, 0), (2, -1), (-1, 2)];
const I32: K = &[(1, 0), (-2, 0), (1, -2), (-2, 1)];
const I30: K = &[(1, 0), (-2, 0), (1, 2), (-2, -1)];
const I03: K = &[(-1, 0), (2, 0), (2, 1), (-1, -2)];
const I02: K = &[(0, -1)];
const I13: K = &[(1, 0)];
const I20: K = &[(0, 1)];
const I31: K = &[(-1, 0)];

/// `KICKS[from][to]` for J/L/S/T/Z/O.
pub static KICKS: [[K; 4]; 4] =
    [[EMPTY, K01, K02, K03], [K10, EMPTY, K12, K13], [K20, K21, EMPTY, K23], [K30, K31, K32, EMPTY]];

/// `I_KICKS[from][to]` for I.
pub static I_KICKS: [[K; 4]; 4] =
    [[EMPTY, I01, I02, I03], [I10, EMPTY, I12, I13], [I20, I21, EMPTY, I23], [I30, I31, I32, EMPTY]];

/// Kick list for a rotation `from → to` (excluding test 0).
#[inline]
pub fn kicks(piece: Piece, from: Rotation, to: Rotation) -> &'static [(i8, i8)] {
    if piece == Piece::I { I_KICKS[from.index()][to.index()] } else { KICKS[from.index()][to.index()] }
}

/// `#isTSpinKick` (`src/engine/index.ts:1092-1107`): the TST / fin kick. True when the rotation id is
/// `03`/`23` and the table entry is `(1, 2)`, or the id is `01`/`21` and the entry is `(-1, 2)`.
/// `kick` is the table entry (dy down) that succeeded; `None` for test 0.
#[inline]
pub fn is_tst_or_fin_kick(from: Rotation, to: Rotation, kick: Option<(i8, i8)>) -> bool {
    let Some((kx, ky)) = kick else { return false };
    use Rotation::*;
    match (from, to) {
        (South, West) | (North, West) => kx == 1 && ky == 2,
        (South, East) | (North, East) => kx == -1 && ky == 2,
        _ => false,
    }
}
