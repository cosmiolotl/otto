//! Fast move generation: every distinct final placement `(cells, spin)` of the current piece (and the hold
//! piece) of a [`PlayerState`], with the last-move spin tag the simulator's lock will use.
//!
//! Algorithm (see `docs/MOVEGEN.md`): per rotation, a column-major bitboard of piece positions
//! ([`bits::Bits`]). The usable map (where the piece fits) is built from the board's empty columns; the
//! reachable set grows from the start position by
//! - a horizontal closure (two sweeps per row segment = any number of 1-cell shifts),
//! - drops: `Sonic` adds the landing position under every reachable position (sonic `softDrop`), `Step`
//!   adds every position below (one-row falls),
//! - rotations: one "kick wave" per (rotation, direction) over the not-yet-rotated frontier, testing the
//!   SRS+ kick list in order and masking out sources that an earlier test already accepted,
//!
//! until nothing changes. Spin outcomes come from how each grounded position is entered (shift / drop /
//! fall → `None`; rotation → [`detect_spin`] of that position, with the T fin/TST flag decided by which
//! kick test succeeded), so they follow the simulator's own spin rules exactly.
//!
//! The bitboard reachability approach (usable maps per rotation, flood fill, masked first-valid-kick
//! waves) follows the design of cobra-movegen (Kixenon) as used by Fusion (MosaicTETR/fusion,
//! `src/smear*`); this file is an independent implementation, no code was copied.

// Column loops index several parallel arrays at once; iterator forms would obscure that.
#![allow(clippy::needless_range_loop)]

mod bits;
pub mod fixtures;
pub mod path;
pub mod perft;

use bits::{Bits, TABLES, W, columns, fill_down, usable};

use crate::board::Board;
use crate::falling::{Falling, Spin, detect_spin};
use crate::kicks::is_tst_or_fin_kick;
use crate::piece::{Piece, Rotation};
use crate::state::{Placement, PlayerState};

pub use path::Pathfinder;
pub use perft::{PerftOpts, perft, perft_board};

/// Reachability model (DECISIONS.md D7). The default is `Sonic`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Mode {
    /// Exactly triangle's `press()` model: 1-cell shifts, DAS, rotations with kicks, **sonic** soft drop,
    /// no gravity. Every placement has a `press()` key path and can be checked by the oracle.
    #[default]
    Sonic,
    /// `Sonic` plus one-row falls: the piece can stop at any height (gravity / finite soft drop). A
    /// superset of `Sonic`. Paths may use the simulator-only [`crate::Key::SoftDropStep`].
    Step,
}

impl Mode {
    pub fn from_name(s: &str) -> Option<Mode> {
        match s {
            "sonic" => Some(Mode::Sonic),
            "step" => Some(Mode::Step),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Mode::Sonic => "sonic",
            Mode::Step => "step",
        }
    }
}

/// Options for [`generate`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct GenOptions {
    pub mode: Mode,
    /// Also generate the hold branch (placements with `hold: true` for the piece that comes out of hold).
    pub hold: bool,
}

impl GenOptions {
    pub const fn new(mode: Mode, hold: bool) -> GenOptions {
        GenOptions { mode, hold }
    }
}

/// Append every distinct placement of `state`'s current piece (and, with `opts.hold`, of the hold piece)
/// to `out`. Placements feed [`PlayerState::apply_placement`] directly.
///
/// Dedupe: one placement per `(final cells, spin)` per branch. Symmetric pieces are reported in a
/// canonical rotation (I/S/Z: North or East; O: North). The hold branch is skipped when hold is locked or
/// the hold slot holds the current piece (the swap would change nothing). No allocation once `out` has
/// enough capacity.
pub fn generate(state: &PlayerState, opts: GenOptions, out: &mut Vec<Placement>) {
    let empty = empty_columns(&state.board);
    let m = solve(&state.board, &empty, state.falling, state.last_spin, opts.mode);
    m.emit(false, out);
    if opts.hold {
        if let Some((start, _)) = hold_start(state) {
            let m = solve(&state.board, &empty, start, state.last_spin, opts.mode);
            m.emit(true, out);
        }
    }
}

/// Placements of one piece from an arbitrary start position (`start_spin` = the `lastSpin` the piece
/// carries, normally `None`). `hold` only sets the flag on the output.
pub fn generate_from(
    board: &Board,
    start: Falling,
    start_spin: Spin,
    mode: Mode,
    hold: bool,
    out: &mut Vec<Placement>,
) {
    let empty = empty_columns(board);
    solve(board, &empty, start, start_spin, mode).emit(hold, out);
}

/// Number of distinct placements of one piece; with `by_cells`, spin variants of the same cells count once.
pub fn count_from(board: &Board, start: Falling, start_spin: Spin, mode: Mode, by_cells: bool) -> u32 {
    let empty = empty_columns(board);
    solve(board, &empty, start, start_spin, mode).count(by_cells)
}

/// Start position of the hold branch (the piece that comes out of hold, at its spawn / clutch position),
/// or `None` if the branch is skipped. The bool is "fits".
pub fn hold_start(state: &PlayerState) -> Option<(Falling, bool)> {
    let hp = state.hold_piece()?;
    if state.hold == Some(state.falling.piece) {
        return None;
    }
    Some(state.spawn_falling(hp))
}

/// Sorted absolute cells of a placement (the dedupe identity of `(cells, spin)`).
pub fn placement_cells(p: &Placement) -> [(i8, i8); 4] {
    p.falling().sorted_cells()
}

/// Canonical rotation for a piece (the one movegen reports for symmetric pieces).
#[inline]
pub fn canonical_rotation(piece: Piece, rot: Rotation) -> Rotation {
    match piece {
        Piece::O => Rotation::North,
        Piece::I | Piece::S | Piece::Z => Rotation::from_u8(rot as u8 & 1),
        _ => rot,
    }
}

#[inline]
fn empty_columns(board: &Board) -> [u64; W] {
    let cols = columns(board);
    let mut e = [0u64; W];
    for c in 0..W {
        e[c] = !cols[c] & bits::ROWS;
    }
    e
}

/// Final outcome masks, folded to canonical rotations.
struct Masks {
    /// `[spin][rot]`: spin 0 None, 1 Mini, 2 Normal.
    m: [[Bits; 4]; 3],
    ncanon: usize,
    piece: Piece,
}

impl Masks {
    fn emit(&self, hold: bool, out: &mut Vec<Placement>) {
        let piece = self.piece;
        let geo = &TABLES.geo[piece.index()];
        for rot in 0..self.ncanon {
            let g = &geo[rot];
            let r = Rotation::from_u8(rot as u8);
            for c in 0..W {
                let mut any = self.m[0][rot].0[c] | self.m[1][rot].0[c] | self.m[2][rot].0[c];
                while any != 0 {
                    let row = any.trailing_zeros();
                    any &= any - 1;
                    let x = c as i8 - g.min_dx;
                    let y = row as i8 + g.max_dy;
                    for (s, spin) in [Spin::None, Spin::Mini, Spin::Normal].into_iter().enumerate() {
                        if self.m[s][rot].0[c] >> row & 1 != 0 {
                            out.push(Placement { piece, hold, rot: r, x, y, spin });
                        }
                    }
                }
            }
        }
    }

    fn count(&self, by_cells: bool) -> u32 {
        let mut n = 0;
        for rot in 0..self.ncanon {
            if by_cells {
                n += self.m[0][rot].or(&self.m[1][rot]).or(&self.m[2][rot]).count();
            } else {
                n += self.m[0][rot].count() + self.m[1][rot].count() + self.m[2][rot].count();
            }
        }
        n
    }
}

/// Horizontal closure: any number of 1-cell shifts inside `u`.
#[inline(always)]
fn hsweep(r: &mut Bits, u: &Bits) {
    for c in 1..W {
        r.0[c] |= r.0[c - 1] & u.0[c];
    }
    for c in (0..W - 1).rev() {
        r.0[c] |= r.0[c + 1] & u.0[c];
    }
}

/// Close `r` under shifts and drops (sonic landings, or every one-row fall in `Step`).
#[inline(always)]
fn close(r: &mut Bits, u: &Bits, g: &Bits, mode: Mode) {
    loop {
        hsweep(r, u);
        let mut changed = 0u64;
        for c in 0..W {
            let f = fill_down(r.0[c], u.0[c]);
            let add = match mode {
                Mode::Sonic => f & g.0[c],
                Mode::Step => f,
            };
            changed |= add & !r.0[c];
            r.0[c] |= add;
        }
        if changed == 0 {
            break;
        }
    }
}

#[inline(always)]
fn box_of(f: &Falling, nrot: usize) -> (usize, usize, usize) {
    let ri = if nrot == 1 { 0 } else { f.rot.index() };
    let g = &TABLES.geo[f.piece.index()][ri];
    (ri, (f.x + g.min_dx) as usize, (f.y - g.max_dy) as usize)
}

fn solve(board: &Board, empty: &[u64; W], start: Falling, start_spin: Spin, mode: Mode) -> Masks {
    let t = &*TABLES;
    let piece = start.piece;
    let pi = piece.index();
    let geo = &t.geo[pi];
    let is_o = piece == Piece::O;
    let nrot = if is_o { 1 } else { 4 };

    let mut u = [Bits::EMPTY; 4];
    let mut g = [Bits::EMPTY; 4];
    for r in 0..nrot {
        u[r] = usable(empty, &geo[r]);
        for c in 0..W {
            g[r].0[c] = u[r].0[c] & !(u[r].0[c] << 1);
        }
    }

    let mut reach = [Bits::EMPTY; 4];
    // Positions entered by a rotation (non-fin test), by a T fin/TST kick, or (blocked start only) by a
    // shift/drop from the start.
    let mut rot_in = [Bits::EMPTY; 4];
    let mut fin_in = [Bits::EMPTY; 4];
    let mut seed_none = [Bits::EMPTY; 4];

    let start_fits = start.fits(board);
    if start_fits {
        let (ri, c, r) = box_of(&start, nrot);
        reach[ri].set(c, r);
    } else {
        // Blocked-out start (only after a hold into a blocked spawn): its successors are the seeds.
        // Legality is checked at the destination only, like the engine.
        for i in 0..4 {
            let mut f = start;
            let moved = match i {
                0 => f.shift(board, -1),
                1 => f.shift(board, 1),
                2 => f.sonic_drop(board),
                _ => mode == Mode::Step && f.step_down(board),
            };
            if moved {
                let (ri, c, r) = box_of(&f, nrot);
                reach[ri].set(c, r);
                seed_none[ri].set(c, r);
            }
        }
        for amount in bits::AMOUNTS {
            let mut f = start;
            if let Some(rot) = f.rotate(board, amount) {
                let (ri, c, r) = box_of(&f, nrot);
                reach[ri].set(c, r);
                if piece == Piece::T && is_tst_or_fin_kick(rot.from, rot.to, rot.kick) {
                    fin_in[ri].set(c, r);
                } else {
                    rot_in[ri].set(c, r);
                }
            }
        }
    }

    let mut done = [Bits::EMPTY; 4];
    loop {
        let mut progressed = false;
        for a in 0..nrot {
            close(&mut reach[a], &u[a], &g[a], mode);
            let frontier = reach[a].andnot(&done[a]);
            if !frontier.any() {
                continue;
            }
            progressed = true;
            done[a] = reach[a];
            if is_o {
                // O rotates in place (test 0 always passes from a legal position): handled at the end.
                continue;
            }
            for k in 0..3 {
                let rt = &t.rot[pi][a][k];
                let b = rt.to as usize;
                let mut rem = frontier;
                for i in 0..rt.n as usize {
                    let (dc, dr) = rt.tr[i];
                    let ok = rem.and(&u[b].shifted(-dc, -dr));
                    if ok.any() {
                        let tgt = ok.shifted(dc, dr);
                        reach[b].or_assign(&tgt);
                        if rt.fin >> i & 1 != 0 {
                            fin_in[b].or_assign(&tgt);
                        } else {
                            rot_in[b].or_assign(&tgt);
                        }
                        rem = rem.andnot(&ok);
                        if !rem.any() {
                            break;
                        }
                    }
                }
            }
        }
        if !progressed {
            break;
        }
    }

    let mut m = [[Bits::EMPTY; 4]; 3];
    for b in 0..nrot {
        let (ub, gb, rb) = (&u[b], &g[b], &reach[b]);
        let mut none = Bits::EMPTY;
        let mut rot = rot_in[b].and(gb);
        if is_o {
            // Any reachable O can "rotate" in place.
            rot.or_assign(&rb.and(gb));
        }
        for c in 0..W {
            let left = if c > 0 { rb.0[c - 1] } else { 0 };
            let right = if c + 1 < W { rb.0[c + 1] } else { 0 };
            let move_in = (left | right) & ub.0[c];
            let drop_in = fill_down(rb.0[c] & !gb.0[c], ub.0[c]);
            none.0[c] = (move_in | drop_in | seed_none[b].0[c]) & gb.0[c];
        }
        if piece == Piece::T {
            let fin = fin_in[b].and(gb);
            let gt = &geo[b];
            let r_ = Rotation::from_u8(b as u8);
            for c in 0..W {
                let mut any = rot.0[c] | fin.0[c];
                while any != 0 {
                    let row = any.trailing_zeros() as usize;
                    any &= any - 1;
                    let f = Falling::new(piece, r_, c as i8 - gt.min_dx, row as i8 + gt.max_dy);
                    if rot.0[c] >> row & 1 != 0 {
                        m[detect_spin(board, &f, false) as usize][b].set(c, row);
                    }
                    if fin.0[c] >> row & 1 != 0 {
                        m[detect_spin(board, &f, true) as usize][b].set(c, row);
                    }
                }
            }
        } else {
            // all-mini+: a rotation into an immobile position is a mini, otherwise no spin.
            for c in 0..W {
                let left = if c > 0 { ub.0[c - 1] } else { 0 };
                let right = if c + 1 < W { ub.0[c + 1] } else { 0 };
                let imm = ub.0[c] & !left & !right & !(ub.0[c] >> 1) & !(ub.0[c] << 1);
                none.0[c] |= rot.0[c] & !imm;
                m[1][b].0[c] |= rot.0[c] & imm;
            }
        }
        m[0][b].or_assign(&none);
    }
    // The start position keeps its own lastSpin if it is hard-dropped without moving.
    if start_fits && start.is_grounded(board) {
        let (ri, c, r) = box_of(&start, nrot);
        m[start_spin as usize][ri].set(c, r);
    }

    // Fold symmetric rotations (same cells ⇔ same box).
    let ncanon = match piece {
        Piece::O => 1,
        Piece::I | Piece::S | Piece::Z => {
            for s in m.iter_mut() {
                let (lo, hi) = s.split_at_mut(2);
                lo[0].or_assign(&hi[0]);
                lo[1].or_assign(&hi[1]);
            }
            2
        }
        _ => 4,
    };
    Masks { m, ncanon, piece }
}
