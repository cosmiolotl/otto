//! Two-player versus on a shared frame clock (RULES.md §12.2, DECISIONS.md D4).
//!
//! Each player locks a piece every `frames_per_piece[p]` frames (that is how PPS is set). Both engines
//! tick in lockstep. When both lock on the same frame, player 0 goes first. Attack left after cancelling
//! goes straight into the opponent's queue with `frame = receiver.frame`, `cid` from one global counter
//! starting at 1, and `gameid = sender + 1`.

use crate::config::Config;
use crate::garbage::IncomingGarbage;
use crate::state::{Key, LockObserver, LockOutcome, Placement, PlayerState};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatchResult {
    /// Player `p` won (the other topped out).
    Win(usize),
    /// No winner (e.g. a piece limit was reached; set by match runners, never by [`Match`] itself).
    Draw,
}

#[derive(Clone, Debug)]
pub struct Match {
    pub players: [PlayerState; 2],
    /// Shared frame clock (= every player's `frame`).
    pub frame: u32,
    pub frames_per_piece: [u32; 2],
    /// Frame of each player's next lock.
    pub next_lock: [u32; 2],
    /// Last cid handed out (the next item gets `next_cid + 1`).
    pub last_cid: u32,
    pub result: Option<MatchResult>,
}

impl Match {
    /// Both players get the same seed (RULES.md §11). Player `p` first locks at frame
    /// `frames_per_piece[p]`.
    pub fn new(seed: i64, cfg: &'static Config, frames_per_piece: [u32; 2]) -> Match {
        assert!(frames_per_piece.iter().all(|&f| f > 0), "frames_per_piece must be > 0");
        Match {
            players: [PlayerState::new(seed, cfg), PlayerState::new(seed, cfg)],
            frame: 0,
            frames_per_piece,
            next_lock: frames_per_piece,
            last_cid: 0,
            result: None,
        }
    }

    pub fn tl(seed: i64, frames_per_piece: [u32; 2]) -> Match {
        Match::new(seed, Config::tl(), frames_per_piece)
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        self.result.is_some()
    }

    /// Advance both players to `frame` (must not be in the past).
    pub fn advance_to(&mut self, frame: u32) {
        assert!(frame >= self.frame);
        let n = frame - self.frame;
        for p in &mut self.players {
            p.tick_n(n);
        }
        self.frame = frame;
    }

    /// The player who locks next (earliest `next_lock`, player 0 on ties), after advancing the clock to
    /// that frame. Returns `None` if the match is over.
    pub fn next_turn(&mut self) -> Option<usize> {
        if self.is_over() {
            return None;
        }
        let p = if self.next_lock[1] < self.next_lock[0] { 1 } else { 0 };
        self.advance_to(self.next_lock[p]);
        Some(p)
    }

    /// Route `amounts` from `from` into every other player's queue.
    pub fn route(&mut self, from: usize, amounts: &[u32]) {
        route(&mut self.players, from, amounts, &mut self.last_cid);
    }

    /// External garbage (e.g. a test injection) into player `to`, `gameid` 0.
    pub fn inject(&mut self, to: usize, amount: u32) {
        self.last_cid += 1;
        let p = &mut self.players[to];
        p.receive_garbage(IncomingGarbage { frame: p.frame, amount, cid: self.last_cid, gameid: 0 });
    }

    /// Player `p` locks `placement` now: apply, route the sent events, schedule the next lock and check
    /// for top-out.
    pub fn play(&mut self, p: usize, placement: Placement) -> LockOutcome {
        self.play_with(p, placement, &mut ())
    }

    pub fn play_with<O: LockObserver>(&mut self, p: usize, placement: Placement, obs: &mut O) -> LockOutcome {
        let out = self.players[p].apply_placement_with(placement, obs);
        self.after_lock(p, &out);
        out
    }

    /// Player `p` presses `keys` now (like `press()`, up to the first `HardDrop`). Returns `None` if the
    /// keys contain no `HardDrop` (the presses still happen).
    pub fn play_keys_with<O: LockObserver>(&mut self, p: usize, keys: &[Key], obs: &mut O) -> Option<LockOutcome> {
        let out = self.players[p].apply_keys_with(keys, obs)?;
        self.after_lock(p, &out);
        Some(out)
    }

    pub fn play_keys(&mut self, p: usize, keys: &[Key]) -> Option<LockOutcome> {
        self.play_keys_with(p, keys, &mut ())
    }

    /// Bookkeeping after player `p` locked (by placement or keys) at the current frame.
    pub fn after_lock(&mut self, p: usize, out: &LockOutcome) {
        self.route(p, out.sent.as_slice());
        self.next_lock[p] = self.frame + self.frames_per_piece[p];
        if out.topout {
            // The match ends at once: an opponent lock scheduled for the same frame never happens.
            self.result = Some(MatchResult::Win(1 - p));
        }
    }
}

/// Deliver `amounts` sent by `from` to every other player: one item per amount at the receiver's current
/// frame, `cid`s counting on from `last_cid`, `gameid = from + 1`.
pub fn route(players: &mut [PlayerState], from: usize, amounts: &[u32], last_cid: &mut u32) {
    for (q, p) in players.iter_mut().enumerate() {
        if q == from {
            continue;
        }
        for &amount in amounts {
            *last_cid += 1;
            p.receive_garbage(IncomingGarbage { frame: p.frame, amount, cid: *last_cid, gameid: from as u32 + 1 });
        }
    }
}
