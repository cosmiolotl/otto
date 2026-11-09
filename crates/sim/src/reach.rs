//! Reference (brute-force) placement enumeration: BFS over `(x, y, rotation, lastSpin)` using exactly the
//! `press()` moves (shift 1, rotate CW/CCW/180 with kicks, sonic drop), then a hard drop.
//!
//! This is slow and simple on purpose. It is the ground truth for the fast `movegen`, and it gives tests
//! and benches legal placements plus a shortest key path for each.
//!
//! [`reachable`] / [`reachable_from`] are the original M2 reference (6 keys, dedupe by
//! `(rot, x, y, spin)`); the trace generator depends on their exact output. [`reachable_with`] adds the
//! options M3 needs: DAS keys (so path lengths match the pathfinder's key set), the simulator-only
//! one-row [`Key::SoftDropStep`] (the `Step` reachability model), and dedupe by final cells.

use std::collections::VecDeque;

use crate::board::Board;
use crate::falling::{Falling, Spin};
use crate::piece::Rotation;
use crate::state::{Key, Placement, PlayerState};

/// One reachable final placement and a shortest key sequence (ending in `HardDrop`) that produces it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reachable {
    pub placement: Placement,
    pub keys: Vec<Key>,
}

/// Options for [`reachable_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ReachOpts {
    /// Also use `dasLeft` / `dasRight` (does not change the placement set, only path lengths).
    pub das: bool,
    /// Also use the one-row [`Key::SoftDropStep`] (the `Step` model: the piece can stop at any height).
    pub step: bool,
    /// Dedupe final placements by `(sorted cells, spin)` instead of `(rot, x, y, spin)`, and skip final
    /// positions that overlap the stack (only possible from a blocked-out spawn).
    pub by_cells: bool,
}

/// BFS key order. It is also the tie-break order between shortest paths: BFS with a FIFO queue and a
/// fixed key order finds, for every state, the lexicographically smallest shortest key sequence.
/// The first six are the M2 order used by [`reachable`].
pub const MOVE_ORDER: [Key; 9] = [
    Key::MoveLeft,
    Key::MoveRight,
    Key::RotateCw,
    Key::RotateCcw,
    Key::Rotate180,
    Key::SoftDrop,
    Key::DasLeft,
    Key::DasRight,
    Key::SoftDropStep,
];

/// The keys used for `opts`, in BFS order.
pub fn moves_for(opts: ReachOpts) -> Vec<Key> {
    MOVE_ORDER
        .iter()
        .copied()
        .filter(|k| match k {
            Key::DasLeft | Key::DasRight => opts.das,
            Key::SoftDropStep => opts.step,
            _ => true,
        })
        .collect()
}

#[inline]
fn idx(f: &Falling, spin: Spin) -> usize {
    // x in -3..13, y in -3..45 → generous bounds.
    let x = (f.x as i32 + 3) as usize;
    let y = (f.y as i32 + 3) as usize;
    ((x * 48 + y) * 4 + f.rot.index()) * 3 + spin as usize
}
const STATES: usize = 16 * 48 * 4 * 3;

/// Apply one movement key to `f`; returns the new position and `lastSpin` if the key did something.
pub fn step(board: &Board, f: Falling, key: Key) -> Option<(Falling, Spin)> {
    let mut g = f;
    match key {
        Key::MoveLeft => g.shift(board, -1).then_some((g, Spin::None)),
        Key::MoveRight => g.shift(board, 1).then_some((g, Spin::None)),
        Key::DasLeft => g.das(board, -1).then_some((g, Spin::None)),
        Key::DasRight => g.das(board, 1).then_some((g, Spin::None)),
        Key::SoftDrop => g.sonic_drop(board).then_some((g, Spin::None)),
        Key::SoftDropStep => g.step_down(board).then_some((g, Spin::None)),
        Key::RotateCw => g.rotate(board, 1).map(|r| (g, r.spin)),
        Key::RotateCcw => g.rotate(board, -1).map(|r| (g, r.spin)),
        Key::Rotate180 => g.rotate(board, 2).map(|r| (g, r.spin)),
        Key::Hold | Key::HardDrop => None,
    }
}

/// All distinct final placements of `piece` starting from `start` with spin `start_spin`.
pub fn reachable_from(board: &Board, start: Falling, start_spin: Spin, hold: bool) -> Vec<Reachable> {
    bfs(board, start, start_spin, hold, &MOVE_ORDER[..6], false)
}

/// [`reachable_from`] with options.
pub fn reachable_from_with(
    board: &Board,
    start: Falling,
    start_spin: Spin,
    hold: bool,
    opts: ReachOpts,
) -> Vec<Reachable> {
    bfs(board, start, start_spin, hold, &moves_for(opts), opts.by_cells)
}

fn bfs(board: &Board, start: Falling, start_spin: Spin, hold: bool, moves: &[Key], by_cells: bool) -> Vec<Reachable> {
    let mut parent: Vec<u32> = vec![u32::MAX; STATES];
    let mut via: Vec<u8> = vec![0; STATES];
    let mut node: Vec<Option<(Falling, Spin)>> = vec![None; STATES];
    let mut q = VecDeque::new();
    let s0 = idx(&start, start_spin);
    parent[s0] = s0 as u32;
    node[s0] = Some((start, start_spin));
    q.push_back(s0);
    let mut out: Vec<Reachable> = Vec::new();
    let mut seen: Vec<(Rotation, i8, i8, Spin)> = Vec::new();
    let mut seen_cells: Vec<([(i8, i8); 4], Spin)> = Vec::new();
    while let Some(i) = q.pop_front() {
        let (f, spin) = node[i].unwrap();
        let mut d = f;
        let fell = d.sonic_drop(board);
        let fspin = if fell { Spin::None } else { spin };
        let fresh = if by_cells {
            let key = (d.sorted_cells(), fspin);
            let fresh = d.fits(board) && !seen_cells.contains(&key);
            if fresh {
                seen_cells.push(key);
            }
            fresh
        } else {
            let key = (d.rot, d.x, d.y, fspin);
            let fresh = !seen.contains(&key);
            if fresh {
                seen.push(key);
            }
            fresh
        };
        if fresh {
            let mut keys = Vec::new();
            let mut j = i;
            while parent[j] as usize != j {
                keys.push(moves[via[j] as usize]);
                j = parent[j] as usize;
            }
            if hold {
                keys.push(Key::Hold);
            }
            keys.reverse();
            keys.push(Key::HardDrop);
            out.push(Reachable {
                placement: Placement { piece: f.piece, hold, rot: d.rot, x: d.x, y: d.y, spin: fspin },
                keys,
            });
        }
        for (m, &k) in moves.iter().enumerate() {
            if let Some((g, s)) = step(board, f, k) {
                let j = idx(&g, s);
                if parent[j] == u32::MAX {
                    parent[j] = i as u32;
                    via[j] = m as u8;
                    node[j] = Some((g, s));
                    q.push_back(j);
                }
            }
        }
    }
    out
}

/// All placements for the current piece and (if hold is available) the hold piece.
///
/// The spawn position inherits the state's `last_spin` (the engine does not clear `lastSpin` on spawn or
/// hold, RULES.md §3.3).
pub fn reachable(state: &PlayerState) -> Vec<Reachable> {
    let mut v = reachable_from(&state.board, state.falling, state.last_spin, false);
    // Swapping with an identical held piece changes nothing, so skip that branch.
    if !state.hold_locked && state.hold != Some(state.falling.piece) {
        let mut s = state.clone();
        if s.hold() {
            v.extend(reachable_from(&s.board, s.falling, s.last_spin, true));
        }
    }
    v
}

/// [`reachable`] with options. With `hold`, the hold branch is included (same skip rule as
/// [`reachable`]: not when hold is locked or the held piece equals the current one).
pub fn reachable_with(state: &PlayerState, opts: ReachOpts, hold: bool) -> Vec<Reachable> {
    let mut v = reachable_from_with(&state.board, state.falling, state.last_spin, false, opts);
    if hold && !state.hold_locked && state.hold != Some(state.falling.piece) {
        let mut s = state.clone();
        if s.hold() {
            v.extend(reachable_from_with(&s.board, s.falling, s.last_spin, true, opts));
        }
    }
    v
}
