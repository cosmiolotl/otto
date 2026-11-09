//! Perft: count the leaf placement sequences of a fixed depth through the queue.
//!
//! - [`perft`] walks real game states ([`PlayerState::apply_placement`]: line clears, attack, garbage,
//!   clutch, block-out). A placement whose lock reports `topout` ends the game, so it has no
//!   continuations (it counts only as a depth-1 leaf).
//! - [`perft_board`] walks bare boards with a fixed cyclic queue (place + clear lines, nothing else), and
//!   a spawn that is blocked has no placements. This is the configuration used to compare with Fusion /
//!   cobra-movegen (`docs/MOVEGEN.md`).
//!
//! Hold (when enabled), per ply: the current piece's placements, plus the hold branch's placements
//! (`hold: true`). The hold branch is skipped when hold is locked or the hold slot holds the current piece.
//! With an empty hold slot, holding uses the next queue piece, so the queue advances by two.
//!
//! Dedupe per ply and branch: `(cells, spin)` by default; with `by_cells`, spin variants of the same cells
//! count (and expand) once.

use crate::board::Board;
use crate::falling::{Falling, Spin};
use crate::piece::Piece;
use crate::state::{Placement, PlayerState};

use super::{GenOptions, Mode, count_from, generate, generate_from, placement_cells};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PerftOpts {
    pub mode: Mode,
    pub hold: bool,
    /// Count each final cell set once, ignoring spin variants.
    pub by_cells: bool,
}

/// Keep the first placement of every `(hold, cells)` group. Movegen emits the spin variants of one
/// position next to each other, and symmetric rotations are already folded into one canonical rotation,
/// so equal cells ⇔ equal `(rot, x, y)` ⇔ adjacent.
fn dedupe_cells(v: &mut Vec<Placement>) {
    v.dedup_by_key(|p| (p.hold, p.rot, p.x, p.y));
    debug_assert!({
        let mut k: Vec<_> = v.iter().map(|p| (p.hold, placement_cells(p))).collect();
        k.sort_unstable();
        k.windows(2).all(|w| w[0] != w[1])
    });
}

/// Leaf count of `depth` placements from `state`.
pub fn perft(state: &PlayerState, depth: u32, opts: PerftOpts) -> u64 {
    let mut bufs: Vec<Vec<Placement>> = (0..depth).map(|_| Vec::with_capacity(256)).collect();
    perft_rec(state, depth, opts, &mut bufs)
}

fn perft_rec(state: &PlayerState, depth: u32, opts: PerftOpts, bufs: &mut [Vec<Placement>]) -> u64 {
    if depth == 0 {
        return 1;
    }
    let (buf, rest) = bufs.split_first_mut().unwrap();
    buf.clear();
    generate(state, GenOptions::new(opts.mode, opts.hold), buf);
    if opts.by_cells {
        dedupe_cells(buf);
    }
    if depth == 1 {
        return buf.len() as u64;
    }
    let mut n = 0;
    for &p in buf.iter() {
        let mut child = state.clone();
        if child.apply_placement(p).topout {
            continue;
        }
        n += perft_rec(&child, depth - 1, opts, rest);
    }
    n
}

/// Bare-board perft with the cyclic queue `queue` (piece `i` = `queue[i % len]`), starting at `queue[0]`
/// with an empty hold slot.
pub fn perft_board(board: &Board, queue: &[Piece], depth: u32, opts: PerftOpts) -> u64 {
    assert!(!queue.is_empty());
    let mut bufs: Vec<Vec<Placement>> = (0..depth).map(|_| Vec::with_capacity(256)).collect();
    board_rec(board, queue, 0, None, depth, opts, &mut bufs)
}

/// `(piece to place, hold slot after, next queue index, hold flag)`.
type Branch = (Piece, Option<Piece>, usize, bool);

fn spawn(board: &Board, piece: Piece) -> Option<Falling> {
    let f = Falling::spawn(piece);
    f.fits(board).then_some(f)
}

fn board_rec(
    board: &Board,
    queue: &[Piece],
    qi: usize,
    hold: Option<Piece>,
    depth: u32,
    opts: PerftOpts,
    bufs: &mut [Vec<Placement>],
) -> u64 {
    if depth == 0 {
        return 1;
    }
    let q = |i: usize| queue[i % queue.len()];
    let cur = q(qi);
    let mut branches: [Option<Branch>; 2] = [Some((cur, hold, qi + 1, false)), None];
    if opts.hold && hold != Some(cur) {
        branches[1] = Some(match hold {
            None => (q(qi + 1), Some(cur), qi + 2, true),
            Some(h) => (h, Some(cur), qi + 1, true),
        });
    }
    if depth == 1 {
        let mut n = 0u64;
        for &(piece, _, _, _) in branches.iter().flatten() {
            if let Some(f) = spawn(board, piece) {
                n += count_from(board, f, Spin::None, opts.mode, opts.by_cells) as u64;
            }
        }
        return n;
    }
    let (buf, rest) = bufs.split_first_mut().unwrap();
    let mut n = 0u64;
    for &(piece, next_hold, next_qi, flag) in branches.iter().flatten() {
        let Some(f) = spawn(board, piece) else { continue };
        buf.clear();
        generate_from(board, f, Spin::None, opts.mode, flag, buf);
        if opts.by_cells {
            dedupe_cells(buf);
        }
        for p in buf.iter() {
            let mut b = *board;
            let cells = p.falling().cells();
            b.place(&cells);
            let lo = cells.iter().map(|c| c.1).min().unwrap();
            let hi = cells.iter().map(|c| c.1).max().unwrap();
            b.clear_lines_in(lo, hi);
            n += board_rec(&b, queue, next_qi, next_hold, depth - 1, opts, rest);
        }
    }
    n
}
