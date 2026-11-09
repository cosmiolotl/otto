//! Per-player state, the lock pipeline, placements and key input (RULES.md §3, §6, §9, §10).

use crate::attack::{ChainState, ClearInfo, Events, attack};
use crate::board::Board;
use crate::config::{Config, Tracker};
use crate::falling::{Falling, Spin};
use crate::garbage::{GarbageQueue, IncomingGarbage};
use crate::piece::{Piece, Rotation};
use crate::queue::PieceQueue;

/// Cumulative counters (`engine.stats`) plus garbage accounting used by property tests.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Stats {
    pub pieces: u32,
    pub lines: u32,
    /// `stats.garbage.attack`: pre-cancel attack.
    pub attack: u32,
    /// `stats.garbage.sent`: post-cancel attack.
    pub sent: u32,
    /// `stats.garbage.cleared`: garbage rows cleared.
    pub cleared: u32,
    /// Incoming lines that entered the board.
    pub garbage_landed: u32,
    /// Incoming lines removed by cancelling (including opener-phase doubling).
    pub garbage_cancelled: u32,
}

/// Key names of triangle's `press()` (RULES.md "Inputs").
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    MoveLeft,
    MoveRight,
    DasLeft,
    DasRight,
    SoftDrop,
    RotateCw,
    RotateCcw,
    Rotate180,
    Hold,
    HardDrop,
    /// **Simulator-only**, not a triangle `press()` key: fall exactly one row if the row below is free
    /// (one gravity / finite-SDF soft-drop step). `movegen` uses it for `Mode::Step` paths. Traces
    /// that contain it cannot be replayed by the oracle.
    SoftDropStep,
}

impl Key {
    pub const ALL: [Key; 11] = [
        Key::MoveLeft,
        Key::MoveRight,
        Key::DasLeft,
        Key::DasRight,
        Key::SoftDrop,
        Key::RotateCw,
        Key::RotateCcw,
        Key::Rotate180,
        Key::Hold,
        Key::HardDrop,
        Key::SoftDropStep,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Key::MoveLeft => "moveLeft",
            Key::MoveRight => "moveRight",
            Key::DasLeft => "dasLeft",
            Key::DasRight => "dasRight",
            Key::SoftDrop => "softDrop",
            Key::RotateCw => "rotateCW",
            Key::RotateCcw => "rotateCCW",
            Key::Rotate180 => "rotate180",
            Key::Hold => "hold",
            Key::HardDrop => "hardDrop",
            Key::SoftDropStep => "softDropStep",
        }
    }

    pub fn from_name(s: &str) -> Option<Key> {
        Key::ALL.into_iter().find(|k| k.name() == s)
    }
}

/// A final placement, as produced by move generation.
///
/// `x`/`y`/`rot` are the piece location (`y` = board row of matrix row 0, see [`Falling`]). The position
/// must be resting (cannot move down) and reachable; `spin` is the last-move spin classification
/// (from the rotation that reached it, or [`Spin::None`] if the last move was a shift/drop).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Placement {
    pub piece: Piece,
    /// Press hold first; `piece` is then the piece that comes out (hold slot or next).
    pub hold: bool,
    pub rot: Rotation,
    pub x: i8,
    pub y: i8,
    pub spin: Spin,
}

impl Placement {
    #[inline]
    pub fn falling(&self) -> Falling {
        Falling::new(self.piece, self.rot, self.x, self.y)
    }
}

/// Everything a lock did (`LockRes` plus a few extras). `Copy`, no heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LockOutcome {
    pub piece: Piece,
    /// Final cells of the locked piece (before line clears), sorted by `(y, x)` so that symmetric rotations
    /// with the same cells give identical outcomes.
    pub cells: [(i8, i8); 4],
    pub lines: u8,
    pub spin: Spin,
    pub all_clear: bool,
    pub garbage_cleared: u8,
    /// Bit `y` set ⇔ row `y` was cleared.
    pub cleared_rows: u64,
    /// `rawGarbage`: events before cancelling.
    pub raw: Events,
    /// `garbage`: events after cancelling (sent to the opponent, in order).
    pub sent: Events,
    /// `surge`: surge total before cancelling.
    pub surge: u32,
    pub broke_b2b: bool,
    /// Incoming lines cancelled by this lock.
    pub cancelled: u32,
    /// Incoming lines that entered the board on this lock.
    pub tanked: u32,
    /// The next piece cannot spawn (block out).
    pub topout: bool,
}

/// Hooks for observers that need more than [`LockOutcome`] (e.g. the replayer's piece-colour board).
/// All methods default to no-ops; `()` implements the trait and compiles to nothing.
pub trait LockObserver {
    #[inline]
    fn placed(&mut self, _piece: Piece, _cells: &[(i8, i8); 4]) {}
    #[inline]
    fn cleared(&mut self, _mask: u64) {}
    #[inline]
    fn garbage_line(&mut self, _column: u8, _cid: u32) {}
}

impl LockObserver for () {}

/// Error from [`PlayerState::apply_placement`] validity checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlacementError {
    HoldLocked,
    WrongPiece { expected: Piece, got: Piece },
    Collides,
    NotResting,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlayerState {
    pub cfg: &'static Config,
    pub board: Board,
    pub falling: Falling,
    pub queue: PieceQueue,
    pub hold: Option<Piece>,
    pub hold_locked: bool,
    pub chain: ChainState,
    /// `lastSpin` (null and "none" are both [`Spin::None`]).
    pub last_spin: Spin,
    /// Last lock cleared lines (enables clutch).
    pub last_was_clear: bool,
    pub garbage: GarbageQueue,
    pub stats: Stats,
    /// Engine frame (number of ticks so far).
    pub frame: u32,
    pub multiplier: Tracker,
    pub cap: Tracker,
    /// Sticky: some lock reported `topout`.
    pub topped_out: bool,
    /// The engine's `ACTION_IHS` state flag (see [`PlayerState::spawn`]). Only ever set by a block-out.
    pub ihs_flag: bool,
}

impl PlayerState {
    /// Fresh player: bag RNG and garbage RNG both seeded with `seed`; the first piece is spawned.
    pub fn new(seed: i64, cfg: &'static Config) -> PlayerState {
        let mut queue = PieceQueue::new(seed);
        let first = queue.pop();
        let mut s = PlayerState {
            cfg,
            board: Board::new(),
            falling: Falling::spawn(first),
            queue,
            hold: None,
            hold_locked: false,
            chain: ChainState::default(),
            last_spin: Spin::None,
            last_was_clear: false,
            garbage: GarbageQueue::new(seed),
            stats: Stats::default(),
            frame: 0,
            multiplier: Tracker::new(cfg.multiplier_value),
            cap: Tracker::new(cfg.garbage_cap),
            topped_out: false,
            ihs_flag: false,
        };
        s.spawn(first, false);
        s
    }

    /// TL S2 player.
    pub fn tl(seed: i64) -> PlayerState {
        PlayerState::new(seed, Config::tl())
    }

    #[inline]
    pub fn current(&self) -> Piece {
        self.falling.piece
    }

    /// Current garbage multiplier `M`.
    #[inline]
    pub fn multiplier(&self) -> f64 {
        self.multiplier.value
    }

    /// `engine.toppedOut`: the falling piece overlaps the stack or is out of bounds.
    #[inline]
    pub fn falling_blocked(&self) -> bool {
        !self.falling.fits(&self.board)
    }

    /// Advance one frame (`tick([])` with no inputs and no gravity): frame++, tick the multiplier and
    /// cap trackers.
    #[inline]
    pub fn tick(&mut self) {
        self.frame += 1;
        let c = self.cfg;
        self.multiplier.tick(self.frame, c.multiplier_increase, c.multiplier_margin);
        self.cap.tick(self.frame, c.garbage_cap_increase, c.garbage_cap_margin);
    }

    pub fn tick_n(&mut self, n: u32) {
        let c = self.cfg;
        let end = self.frame + n;
        // Before both margins nothing but the frame counter changes.
        let quiet_until = c.multiplier_margin.min(c.garbage_cap_margin).min(end);
        if self.frame < quiet_until {
            self.frame = quiet_until;
        }
        while self.frame < end {
            self.tick();
        }
    }

    /// Add incoming garbage (`receiveGarbage`).
    #[inline]
    pub fn receive_garbage(&mut self, item: IncomingGarbage) {
        self.garbage.receive(item);
    }

    /// `initiatePiece` (`src/engine/index.ts:938-1006`) with IRS/IHS "tap" and no 20G: spawn, then the
    /// block-out check with clutch.
    ///
    /// Engine quirk reproduced: on block-out the engine does `state ^= ACTION_IHS` (an XOR, so it *sets*
    /// the flag), and the flag survives until a later spawn that is not blocked out, which clears it and
    /// calls `hold()`. That hold is silently refused when hold is locked (spawns that come from a hold).
    #[inline]
    fn spawn(&mut self, piece: Piece, is_hold: bool) {
        if !is_hold {
            self.hold_locked = false;
        }
        let (f, fits) = self.spawn_falling(piece);
        self.falling = f;
        if fits {
            if self.ihs_flag {
                self.spawn_ihs();
            }
            return;
        }
        self.ihs_flag = !self.ihs_flag;
    }

    /// Where `piece` would spawn on the current board: the spawn position, or clutch-raised after a
    /// clearing lock. Also returns whether it fits there; a blocked-out piece stays at its spawn position,
    /// overlapping the stack. Pure. Used by [`Self::spawn`] and by movegen for the hold branch.
    #[inline]
    pub fn spawn_falling(&self, piece: Piece) -> (Falling, bool) {
        let mut f = Falling::spawn(piece);
        if f.fits(&self.board) {
            return (f, true);
        }
        if self.last_was_clear && self.cfg.clutch {
            let y0 = f.y;
            while (f.y as usize) < crate::board::BOARD_HEIGHT {
                f.y += 1;
                if f.fits(&self.board) {
                    return (f, true);
                }
            }
            f.y = y0;
        }
        (f, false)
    }

    /// The piece a `hold` press would bring out (the hold slot, or the next queue piece), or `None` if
    /// hold is locked.
    #[inline]
    pub fn hold_piece(&self) -> Option<Piece> {
        if self.hold_locked { None } else { Some(self.hold.unwrap_or_else(|| self.queue.peek(0))) }
    }

    #[cold]
    fn spawn_ihs(&mut self) {
        self.ihs_flag = false;
        self.hold();
    }

    /// `hold()` via `press("hold")`. Returns false if hold is locked.
    pub fn hold(&mut self) -> bool {
        if self.hold_locked {
            return false;
        }
        self.hold_locked = true;
        let cur = self.falling.piece;
        match self.hold.replace(cur) {
            Some(h) => self.spawn(h, true),
            None => {
                let p = self.queue.pop();
                self.spawn(p, true);
            }
        }
        self.hold_locked = true;
        true
    }

    /// One `press(key)`. Movement keys return whether the piece moved; rotations whether they
    /// succeeded; `HardDrop` locks and returns `Some(outcome)`.
    pub fn press_with<O: LockObserver>(&mut self, key: Key, obs: &mut O) -> (bool, Option<LockOutcome>) {
        let b = &self.board;
        let moved = match key {
            Key::MoveLeft => self.falling.shift(b, -1),
            Key::MoveRight => self.falling.shift(b, 1),
            Key::DasLeft => self.falling.das(b, -1),
            Key::DasRight => self.falling.das(b, 1),
            Key::SoftDrop => self.falling.sonic_drop(b),
            Key::SoftDropStep => self.falling.step_down(b),
            Key::RotateCw | Key::RotateCcw | Key::Rotate180 => {
                let amount = match key {
                    Key::RotateCw => 1,
                    Key::RotateCcw => -1,
                    _ => 2,
                };
                return match self.falling.rotate(b, amount) {
                    Some(r) => {
                        self.last_spin = r.spin;
                        (true, None)
                    }
                    None => (false, None), // failed rotation keeps lastSpin
                };
            }
            Key::Hold => return (self.hold(), None),
            Key::HardDrop => {
                if self.falling.sonic_drop(b) {
                    self.last_spin = Spin::None;
                }
                return (true, Some(self.lock_with(true, obs)));
            }
        };
        if moved {
            self.last_spin = Spin::None;
        }
        (moved, None)
    }

    pub fn press(&mut self, key: Key) -> (bool, Option<LockOutcome>) {
        self.press_with(key, &mut ())
    }

    /// Press keys in order like triangle's `press()`, stopping at the first `HardDrop`. Returns the lock
    /// outcome, or `None` if the keys contain no `HardDrop`.
    pub fn apply_keys_with<O: LockObserver>(&mut self, keys: &[Key], obs: &mut O) -> Option<LockOutcome> {
        for &k in keys {
            if let (_, Some(out)) = self.press_with(k, obs) {
                return Some(out);
            }
        }
        None
    }

    pub fn apply_keys(&mut self, keys: &[Key]) -> Option<LockOutcome> {
        self.apply_keys_with(keys, &mut ())
    }

    /// Lock a placement: optional hold, then put the piece at `(x, y, rot)` with `lastSpin = spin` and
    /// lock it as a hard drop. No validation beyond `debug_assert`s (see [`Self::check_placement`]).
    #[inline]
    pub fn apply_placement(&mut self, p: Placement) -> LockOutcome {
        if p.hold {
            let ok = self.hold();
            debug_assert!(ok, "hold while hold-locked");
        }
        debug_assert_eq!(self.falling.piece, p.piece, "placement piece mismatch");
        self.falling = p.falling();
        debug_assert!(self.falling.fits(&self.board), "placement collides");
        debug_assert!(self.falling.is_grounded(&self.board), "placement not resting");
        self.last_spin = p.spin;
        self.lock_with(true, &mut ())
    }

    /// Validate a placement against the current state without changing it.
    pub fn check_placement(&self, p: &Placement) -> Result<(), PlacementError> {
        let expected = if p.hold {
            if self.hold_locked {
                return Err(PlacementError::HoldLocked);
            }
            self.hold.unwrap_or_else(|| self.queue.peek(0))
        } else {
            self.falling.piece
        };
        if expected != p.piece {
            return Err(PlacementError::WrongPiece { expected, got: p.piece });
        }
        let f = p.falling();
        if !f.fits(&self.board) {
            return Err(PlacementError::Collides);
        }
        if !f.is_grounded(&self.board) {
            return Err(PlacementError::NotResting);
        }
        Ok(())
    }

    /// `#lock(hard)` (`src/engine/index.ts:1368-1635`).
    pub fn lock_with<O: LockObserver>(&mut self, hard: bool, obs: &mut O) -> LockOutcome {
        let cfg = self.cfg;
        self.hold_locked = false;

        let piece = self.falling.piece;
        let cells = sort_cells(self.falling.cells());
        self.board.place(&cells);
        obs.placed(piece, &cells);

        let sh = self.falling.shape();
        let cleared = self.board.clear_lines_in(self.falling.y - sh.max_dy, self.falling.y - sh.min_dy);
        if cleared.lines > 0 {
            obs.cleared(cleared.mask);
        }
        let all_clear = cleared.lines > 0 && self.board.is_empty();
        self.stats.cleared += cleared.garbage as u32;

        let spin = self.last_spin;
        let info = ClearInfo { lines: cleared.lines, spin, all_clear, garbage_cleared: cleared.garbage };
        let res = attack(&info, &mut self.chain, cfg, self.multiplier.value);
        self.stats.attack += res.events.total();

        let mut sent = Events::new();
        let mut cancelled = 0;
        let mut tanked = 0;
        if cleared.lines > 0 {
            self.last_was_clear = true;
            let ev = res.events.as_slice();
            let mut i = 0;
            while i < ev.len() {
                let (left, removed) = self.garbage.cancel(ev[i], self.stats.pieces, cfg);
                cancelled += removed;
                if left > 0 {
                    // Queue is empty now: this remainder and every later event are sent unchanged.
                    sent.push(left);
                    for &e in &ev[i + 1..] {
                        sent.push(e);
                    }
                    break;
                }
                i += 1;
            }
        } else {
            self.last_was_clear = false;
            let board = &mut self.board;
            tanked = self.garbage.tank(self.frame, self.cap.value, hard, cfg, |col, cid| {
                board.insert_garbage(col);
                obs.garbage_line(col, cid);
            });
        }

        let next = self.queue.pop();
        self.spawn(next, false);
        self.last_spin = Spin::None;
        let topout = !self.falling.fits(&self.board);
        if topout {
            self.topped_out = true;
        }

        self.stats.sent += sent.total();
        self.stats.pieces += 1;
        self.stats.lines += cleared.lines as u32;
        self.stats.garbage_landed += tanked;
        self.stats.garbage_cancelled += cancelled;

        LockOutcome {
            piece,
            cells,
            lines: cleared.lines,
            spin,
            all_clear,
            garbage_cleared: cleared.garbage,
            cleared_rows: cleared.mask,
            raw: res.events,
            sent,
            surge: res.surge,
            broke_b2b: res.broke_b2b,
            cancelled,
            tanked,
            topout,
        }
    }
}

/// Sort 4 cells by `(y, x)`: a branchless 5-comparator sorting network on packed `u16` keys.
#[inline(always)]
fn sort_cells(c: [(i8, i8); 4]) -> [(i8, i8); 4] {
    #[inline(always)]
    fn key(c: (i8, i8)) -> u16 {
        (c.1 as u8 as u16) << 8 | c.0 as u8 as u16
    }
    // i8 → u8 keeps the order for the non-negative coordinates of a legal piece.
    let mut k = [key(c[0]), key(c[1]), key(c[2]), key(c[3])];
    #[inline(always)]
    fn cs(k: &mut [u16; 4], i: usize, j: usize) {
        let (a, b) = (k[i], k[j]);
        k[i] = a.min(b);
        k[j] = a.max(b);
    }
    cs(&mut k, 0, 1);
    cs(&mut k, 2, 3);
    cs(&mut k, 0, 2);
    cs(&mut k, 1, 3);
    cs(&mut k, 1, 2);
    k.map(|v| (v as u8 as i8, (v >> 8) as u8 as i8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplier_is_accumulated_per_frame() {
        let mut s = PlayerState::tl(1);
        s.tick_n(10_800);
        assert_eq!(s.multiplier(), 1.0);
        s.tick_n(3_600);
        let mut v = 1.0f64;
        for _ in 0..3_600 {
            v += 0.008 / 60.0;
        }
        assert_eq!(s.multiplier().to_bits(), v.to_bits());
        assert_eq!(s.frame, 14_400);
        assert!((s.multiplier() - 1.48).abs() < 1e-9); // ~ 1 + 60 s × 0.008
    }

    #[test]
    fn hold_rules() {
        let mut s = PlayerState::tl(1_300_715_353); // I O J Z L S T ...
        assert_eq!(s.current(), Piece::I);
        assert!(s.hold());
        assert_eq!((s.current(), s.hold), (Piece::O, Some(Piece::I)));
        assert!(!s.hold(), "hold is locked until the next lock");
        let out = s.apply_keys(&[Key::HardDrop]).unwrap();
        assert_eq!(out.piece, Piece::O);
        assert!(!s.hold_locked);
        assert_eq!(s.current(), Piece::J);
        assert!(s.hold());
        assert_eq!((s.current(), s.hold), (Piece::I, Some(Piece::J)));
        assert_eq!(s.queue.peek(0), Piece::Z, "a swap does not touch the queue");
    }

    #[test]
    fn clutch_only_after_a_clear() {
        // Spawn rows 21-22 blocked at (4,21); a lock that clears lines lets the next piece move up.
        let mut rows = [0u16; 40];
        rows[21] = 1 << 4;
        let mut s = PlayerState::tl(1_300_715_353);
        s.board = Board::from_rows(&rows, 0);
        s.last_was_clear = true;
        s.spawn(Piece::T, false);
        assert!(s.falling.fits(&s.board));
        assert_eq!(s.falling.y, 23);
        s.last_was_clear = false;
        s.spawn(Piece::T, false);
        assert!(!s.falling.fits(&s.board));
        assert_eq!(s.falling.y, 22);
        assert!(s.ihs_flag, "block-out toggles the IHS flag on");
    }

    #[test]
    fn blocked_out_hold_triggers_auto_hold_on_next_spawn() {
        // A block at (6,21) blocks only the I spawn.
        let mut rows = [0u16; 40];
        rows[21] = 1 << 6;
        let mut s = PlayerState::tl(1_300_715_353); // I O J Z L S T
        assert!(s.hold()); // hold I, O comes out
        s.apply_keys(&[Key::DasLeft, Key::HardDrop]).unwrap(); // O locks, J spawns
        s.board = Board::from_rows(
            &{
                let mut r = *s.board.rows();
                r[21] |= rows[21];
                r
            },
            0,
        );
        assert_eq!((s.current(), s.hold), (Piece::J, Some(Piece::I)));
        assert!(s.hold()); // I comes out of hold and is blocked out: not a top-out, flag set
        assert!(s.falling_blocked());
        assert!(s.ihs_flag);
        let out = s.apply_keys(&[Key::HardDrop]).unwrap(); // the I falls from its blocked spawn and locks
        assert!(!out.topout);
        // Next piece Z spawns fine, then the pending IHS holds it: J comes back out of hold.
        assert!(!s.ihs_flag);
        assert_eq!((s.current(), s.hold, s.hold_locked), (Piece::J, Some(Piece::Z), true));
    }
}
