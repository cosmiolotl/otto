//! Pathfinder: the shortest key sequence for a placement.
//!
//! **Shortest** = fewest key presses, counting every key including the final `hardDrop` (and the leading
//! `hold` for hold placements). Keys, in tie-break order: `moveLeft`, `moveRight`, `rotateCW`,
//! `rotateCCW`, `rotate180`, `softDrop` (sonic), `dasLeft`, `dasRight`, plus the simulator-only
//! `softDropStep` in [`Mode::Step`].
//! **Ties** are broken lexicographically in that key order ([`crate::reach::MOVE_ORDER`]): a FIFO BFS
//! that expands keys in a fixed order reaches every state first along its lexicographically smallest
//! shortest path, and the first popped state whose hard drop gives the target is the answer. `hold`, if
//! needed, is always the first key; `hardDrop` is always the last.
//!
//! The search state is `(x, y, rotation, lastSpin)`, so a spin target is only matched by a path whose
//! last move is the rotation that produces that spin.

use crate::board::Board;
use crate::falling::{Falling, Spin};
use crate::piece::Rotation;
use crate::reach::{MOVE_ORDER, step};
use crate::state::{Key, Placement, PlayerState};

use super::{Mode, canonical_rotation, hold_start};

// x in -3..13 (16), y in -3..45 (48), 4 rotations, 3 spins.
const NX: usize = 16;
const NY: usize = 48;
const STATES: usize = NX * NY * 4 * 3;

#[inline]
fn idx(f: &Falling, spin: Spin) -> usize {
    let x = (f.x as i32 + 3) as usize;
    let y = (f.y as i32 + 3) as usize;
    ((x * NY + y) * 4 + f.rot.index()) * 3 + spin as usize
}

#[inline]
fn decode(piece: crate::piece::Piece, i: usize) -> (Falling, Spin) {
    let spin = [Spin::None, Spin::Mini, Spin::Normal][i % 3];
    let rot = Rotation::from_u8(((i / 3) % 4) as u8);
    let y = ((i / 12) % NY) as i8 - 3;
    let x = (i / (12 * NY)) as i8 - 3;
    (Falling::new(piece, rot, x, y), spin)
}

/// Identity of a final placement: canonical rotation, box bottom-left, spin.
#[inline]
fn target_key(f: &Falling, spin: Spin) -> (Rotation, i8, i8, Spin) {
    let sh = f.shape();
    (canonical_rotation(f.piece, f.rot), f.x + sh.min_dx, f.y - sh.max_dy, spin)
}

/// Reusable BFS buffers (no allocation per query after construction, except growing `keys`).
pub struct Pathfinder {
    stamp: Box<[u32]>,
    parent: Box<[u16]>,
    via: Box<[u8]>,
    queue: Box<[u16]>,
    epoch: u32,
}

impl Default for Pathfinder {
    fn default() -> Self {
        Self::new()
    }
}

impl Pathfinder {
    pub fn new() -> Pathfinder {
        Pathfinder {
            stamp: vec![0; STATES].into_boxed_slice(),
            parent: vec![0; STATES].into_boxed_slice(),
            via: vec![0; STATES].into_boxed_slice(),
            queue: vec![0; STATES].into_boxed_slice(),
            epoch: 0,
        }
    }

    fn keys_for(mode: Mode) -> &'static [Key] {
        match mode {
            Mode::Sonic => &MOVE_ORDER[..8],
            Mode::Step => &MOVE_ORDER[..],
        }
    }

    /// Shortest path from `start` (carrying `start_spin`) to a position whose hard drop locks `target`'s
    /// cells with `target.spin`. Appends the moves and the final `HardDrop` to `keys` (no `Hold`).
    /// Returns false (and leaves `keys` unchanged) if the target is unreachable.
    pub fn find_from(
        &mut self,
        board: &Board,
        start: Falling,
        start_spin: Spin,
        target: &Placement,
        mode: Mode,
        keys: &mut Vec<Key>,
    ) -> bool {
        let want = target_key(&target.falling(), target.spin);
        let moves = Self::keys_for(mode);
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
        let epoch = self.epoch;
        let piece = start.piece;
        let s0 = idx(&start, start_spin);
        self.stamp[s0] = epoch;
        self.parent[s0] = s0 as u16;
        self.queue[0] = s0 as u16;
        let (mut head, mut tail) = (0usize, 1usize);
        while head < tail {
            let i = self.queue[head] as usize;
            head += 1;
            let (f, spin) = decode(piece, i);
            let mut d = f;
            let fell = d.sonic_drop(board);
            let fspin = if fell { Spin::None } else { spin };
            if target_key(&d, fspin) == want && d.fits(board) {
                let at = keys.len();
                keys.push(Key::HardDrop);
                let mut j = i;
                while self.parent[j] as usize != j {
                    keys.push(moves[self.via[j] as usize]);
                    j = self.parent[j] as usize;
                }
                keys[at..].reverse();
                return true;
            }
            for (m, &k) in moves.iter().enumerate() {
                if let Some((g, s)) = step(board, f, k) {
                    let j = idx(&g, s);
                    if self.stamp[j] != epoch {
                        self.stamp[j] = epoch;
                        self.parent[j] = i as u16;
                        self.via[j] = m as u8;
                        self.queue[tail] = j as u16;
                        tail += 1;
                    }
                }
            }
        }
        false
    }

    /// Shortest path for a placement of `state` (as produced by [`super::generate`]): `Hold` first if
    /// `target.hold`, then moves, then `HardDrop`. Appends to `keys`; false if unreachable.
    pub fn find(&mut self, state: &PlayerState, target: &Placement, mode: Mode, keys: &mut Vec<Key>) -> bool {
        if target.hold {
            let Some((start, _)) = hold_start(state) else { return false };
            if start.piece != target.piece {
                return false;
            }
            let at = keys.len();
            keys.push(Key::Hold);
            if self.find_from(&state.board, start, state.last_spin, target, mode, keys) {
                return true;
            }
            keys.truncate(at);
            false
        } else {
            if state.falling.piece != target.piece {
                return false;
            }
            self.find_from(&state.board, state.falling, state.last_spin, target, mode, keys)
        }
    }
}
