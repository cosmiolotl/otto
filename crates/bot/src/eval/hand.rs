//! The hand-written evaluation over [`Weights`] (see `docs/BOT.md` for every term), with the release
//! controller that sets its mode once per move.

use tetris_sim::PlayerState;

use super::board::{BoardFeatures, board_features};
use super::{Evaluator, PathAcc};
use crate::controller::burst::Burst;
use crate::controller::{self, Mode, Plan};
use crate::obs::Observation;
use crate::weights::Weights;

#[derive(Debug)]
pub struct HandEval {
    pub weights: Weights,
    /// The current move's plan (BUILD with no threat until [`Evaluator::begin`] runs the controller).
    plan: Plan,
    burst: Burst,
}

impl HandEval {
    pub fn new(weights: Weights) -> HandEval {
        HandEval { weights, plan: Plan::default(), burst: Burst::default() }
    }

    /// The shape/setup part.
    pub fn board_score(&self, f: &BoardFeatures) -> f32 {
        let w = &self.weights;
        let maxh = f.max_height as f32;
        let over = (maxh - w.height_cap).max(0.0);
        let edge = (f.well_col == 0 || f.well_col == 9) && f.well_depth > 0;
        w.holes * f.holes as f32
            + w.hole_rows * f.hole_rows as f32
            + w.covered * f.covered as f32
            + w.bump * f.bump as f32
            + w.bump_sq * f.bump_sq as f32
            + w.max_height * maxh
            + w.avg_height * f.sum_height as f32 * 0.1
            + w.over_cap * over * over
            + w.row_trans * f.row_trans as f32
            + w.col_trans * f.col_trans as f32
            + w.well_depth * (f.well_depth as f32).min(w.well_cap)
            + w.well_edge * edge as u8 as f32
            + w.tslot * f.tslot
            + w.quad_ready * f.quad_ready as f32
            + w.pc_ready * f.pc_ready as u8 as f32
    }
}

impl Evaluator for HandEval {
    fn begin(&mut self, obs: &Observation) -> Plan {
        self.plan = if self.weights.controller {
            controller::decide(obs, &self.weights, &mut self.burst)
        } else {
            Plan::default()
        };
        self.plan
    }

    #[inline]
    fn eval(&self, s: &PlayerState, path: &PathAcc) -> f32 {
        let w = &self.weights;
        let Plan { mode, opp_burst, opp_fill } = self.plan;
        let f = board_features(&s.board);
        let mut v = self.board_score(&f);

        let b2b = s.chain.b2b;
        if b2b >= 0 {
            let lvl = (b2b + 1) as f32;
            v += w.b2b_active
                + w.b2b_level * lvl.min(w.reserve_target)
                + w.b2b_excess * (lvl - w.reserve_target).max(0.0);
        }
        if s.chain.combo >= 0 {
            v += w.combo * (s.chain.combo + 1) as f32;
        }

        v += w.attack * (1.0 + w.pressure * opp_fill) * path.attack
            + w.sent * path.sent
            + w.cancelled * path.cancelled
            + w.tanked * path.tanked
            + w.b2b_break * path.breaks
            + w.waste * path.waste
            + w.t_waste * path.t_waste;
        match mode {
            Mode::Build => {}
            Mode::Kill => v += w.kill_attack * path.attack,
            Mode::Cancel => v += w.cancel_bonus * (path.cancelled - path.tanked),
        }

        // Garbage still queued. With the controller on, the opponent's burst estimate is a spike that may
        // come at any time.
        let pending = s.garbage.size() as f32;
        let spike = w.burst_danger * opp_burst;
        let d = (f.max_height as f32 + pending + spike - w.danger_level).max(0.0);
        v += w.pending * pending + w.danger_sq * d * d;
        v
    }

    fn discount(&self) -> f32 {
        self.weights.discount
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::Board;

    #[test]
    fn holes_and_height_are_bad_tslots_good() {
        let e = HandEval::new(Weights::default());
        let mut s = PlayerState::tl(1);
        let base = e.eval(&s, &PathAcc::default());
        s.board = Board::from_ascii(
            "
            #.........
            .#########
            ",
        );
        let holey = e.eval(&s, &PathAcc::default());
        assert!(holey < base);
        s.board = Board::from_ascii(
            "
            ..#.......
            ##...#####
            ###.######
            ",
        );
        let slot = e.board_score(&board_features(&s.board));
        let no_roof = Board::from_rows(&[s.board.row(0), s.board.row(1)], 0);
        let flat = e.board_score(&board_features(&no_roof));
        assert!(slot > flat, "the TSD slot ({slot}) is worth more than the same board without the roof ({flat})");
    }
}
