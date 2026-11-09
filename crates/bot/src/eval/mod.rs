//! Evaluation: the [`Evaluator`] trait the search scores nodes with, the terms accumulated along a search
//! path ([`PathAcc`]), and the two evaluators: the hand-written [`HandEval`] and the value network
//! [`NnueEval`], with [`AnyEval`] to choose one at run time.

pub mod board;
mod hand;
mod nnue;

use tetris_sim::{LockOutcome, Piece, PlayerState};

use crate::controller::Plan;
use crate::obs::Observation;

pub use hand::HandEval;
pub use nnue::NnueEval;

/// Scores search nodes. Higher is better. [`Evaluator::eval`] is called once per generated child, so it
/// must be cheap.
pub trait Evaluator {
    /// Called once per move, before the search, with what the bot sees. Returns the move's plan (the
    /// release controller's mode and threat estimates; the default plan when there is no controller).
    fn begin(&mut self, obs: &Observation) -> Plan;

    fn eval(&self, s: &PlayerState, path: &PathAcc) -> f32;

    /// Per-ply discount of the path terms.
    fn discount(&self) -> f32 {
        1.0
    }
}

/// Terms accumulated along the search path from the root (each ply's contribution is multiplied by
/// `discount^ply`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathAcc {
    /// Attack generated (pre-cancel `raw` events, includes surge).
    pub attack: f32,
    /// Attack sent (post-cancel).
    pub sent: f32,
    pub cancelled: f32,
    pub tanked: f32,
    /// B2B chains broken (only counted when a chain was active).
    pub breaks: f32,
    /// Surge released.
    pub surge: f32,
    /// Lines cleared for no attack and no garbage rows.
    pub waste: f32,
    /// T pieces locked without a T-spin clear.
    pub t_waste: f32,
    /// `discount^ply` for the next ply.
    pub scale: f32,
    /// Lines sent (post-cancel), not discounted: what the opponent's garbage meter gained along the path.
    pub sent_lines: u32,
}

impl Default for PathAcc {
    fn default() -> Self {
        PathAcc {
            attack: 0.0,
            sent: 0.0,
            cancelled: 0.0,
            tanked: 0.0,
            breaks: 0.0,
            surge: 0.0,
            waste: 0.0,
            t_waste: 0.0,
            scale: 1.0,
            sent_lines: 0,
        }
    }
}

impl PathAcc {
    /// Add one lock. `b2b_before` is the chain counter before the lock.
    #[inline]
    pub fn push(&mut self, out: &LockOutcome, b2b_before: i32, discount: f32) {
        let s = self.scale;
        self.attack += s * out.raw.total() as f32;
        self.sent += s * out.sent.total() as f32;
        self.sent_lines += out.sent.total();
        self.cancelled += s * out.cancelled as f32;
        self.tanked += s * out.tanked as f32;
        if out.broke_b2b && b2b_before >= 0 {
            self.breaks += s;
            self.surge += s * out.surge as f32;
        }
        if out.lines > 0 && out.raw.is_empty() && out.garbage_cleared == 0 {
            self.waste += s * out.lines as f32;
        }
        if out.piece == Piece::T && !(out.lines > 0 && out.spin.is_spin()) {
            self.t_waste += s;
        }
        self.scale *= discount;
    }
}

/// Hand-written or network evaluation, chosen at run time (one bot type for every roster entry).
#[derive(Debug)]
pub enum AnyEval {
    Hand(Box<HandEval>),
    Nnue(Box<NnueEval>),
}

impl Evaluator for AnyEval {
    fn begin(&mut self, obs: &Observation) -> Plan {
        match self {
            AnyEval::Hand(e) => e.begin(obs),
            AnyEval::Nnue(e) => e.begin(obs),
        }
    }

    #[inline]
    fn eval(&self, s: &PlayerState, path: &PathAcc) -> f32 {
        match self {
            AnyEval::Hand(e) => e.eval(s, path),
            AnyEval::Nnue(e) => e.eval(s, path),
        }
    }

    fn discount(&self) -> f32 {
        match self {
            AnyEval::Hand(e) => e.discount(),
            AnyEval::Nnue(e) => e.discount(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::{Board, Piece};

    #[test]
    fn path_acc_counts_breaks_only_with_an_active_chain() {
        // An I clears a single: no chain before, so no break; then the same with b2b 5 releases surge.
        let mut fresh = PlayerState::tl(1);
        fresh.board = Board::from_ascii("#########.");
        fresh.falling = fresh.spawn_falling(Piece::I).0;
        let mut buf = Vec::new();
        tetris_sim::movegen::generate(&fresh, Default::default(), &mut buf);
        let p =
            buf.iter().copied().find(|p| fresh.clone().apply_placement(*p).lines == 1).expect("I clears the single");
        let mut a = PathAcc::default();
        let mut c = fresh.clone();
        let out = c.apply_placement(p);
        a.push(&out, fresh.chain.b2b, 0.9);
        assert_eq!(a.breaks, 0.0);
        assert_eq!(a.waste, 1.0);
        let mut chained = fresh.clone();
        chained.chain.b2b = 5;
        let mut b = PathAcc::default();
        let out = chained.clone().apply_placement(p);
        b.push(&out, 5, 0.9);
        assert_eq!(b.breaks, 1.0);
        assert_eq!(b.surge, 5.0);
        assert_eq!(b.attack, 5.0);
        assert!((b.scale - 0.9).abs() < 1e-6);
    }
}
