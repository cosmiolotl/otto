//! Release controller: picks KILL / CANCEL / BUILD once per move, before the search (`docs/PLAN.md` M4).
//!
//! 1. **KILL** if `kill_margin = my_burst - my_pending + opp_pending - opp_absorb > kill_threshold`,
//!    where `opp_absorb = opp_margin + opp_burst_scale · opp_burst`: my best known-queue release, after
//!    cancelling what is coming at me, plus what is already queued at the opponent, beats their room to
//!    the top plus what they can cancel with.
//! 2. **CANCEL** if garbage is pending and `danger = sigmoid((my_pending + opp_burst_scale · opp_burst -
//!    my_margin) / 3) > cancel_threshold`.
//! 3. **BUILD** otherwise.
//!
//! The mode only changes the hand-written evaluation (extra weight on path attack, or on
//! cancelled-minus-tanked), so the search code is the same in every mode.

pub mod burst;

use tetris_sim::board::VISIBLE_HEIGHT;

use crate::obs::Observation;
use crate::weights::Weights;
use burst::Burst;

/// Placements and beam width of the `my_burst(k)` sub-search.
pub const BURST_DEPTH: usize = 4;
pub const BURST_WIDTH: usize = 6;

/// Release-controller mode (chosen once per move, before the search).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Mode {
    /// Default: full eval, under the height cap, building reserve up to the target.
    #[default]
    Build,
    /// The best release beats what the opponent can absorb: maximise attack now.
    Kill,
    /// Incoming garbage is dangerous: cancel it.
    Cancel,
}

impl Mode {
    pub const COUNT: usize = 3;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            Mode::Build => "BUILD",
            Mode::Kill => "KILL",
            Mode::Cancel => "CANCEL",
        }
    }
}

/// What the controller decided for one move: the mode, and the two threat estimates the evaluation uses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Plan {
    pub mode: Mode,
    /// `my_burst(k)` of the opponent: the most attack they can generate in their known queue (lines).
    pub opp_burst: f32,
    /// How full the opponent is: `clamp((opp_height + opp_pending - 8) / 12, 0, 1)` (scales the value of
    /// attack by `1 + pressure · opp_fill`).
    pub opp_fill: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn decide(obs: &Observation, w: &Weights, burst: &mut Burst) -> Plan {
    let my_burst = burst.of_view(&obs.me, BURST_DEPTH, BURST_WIDTH, obs.my_fpp);
    let opp_burst = burst.of_view(&obs.opp, BURST_DEPTH, BURST_WIDTH, obs.opp_fpp);
    let my_pending = obs.me.pending() as f32;
    let opp_pending = obs.opp.pending() as f32;
    let my_margin = VISIBLE_HEIGHT as f32 - obs.me.board.top() as f32;
    let opp_margin = VISIBLE_HEIGHT as f32 - obs.opp.board.top() as f32;
    let opp_absorb = opp_margin + w.opp_burst_scale * opp_burst;
    let kill_margin = my_burst - my_pending + opp_pending - opp_absorb;
    let danger = sigmoid((my_pending + w.opp_burst_scale * opp_burst - my_margin) / 3.0);
    let opp_fill = ((obs.opp.board.top() as f32 + opp_pending - 8.0) / 12.0).clamp(0.0, 1.0);
    let mode = if kill_margin > w.kill_threshold {
        Mode::Kill
    } else if my_pending > 0.0 && danger > w.cancel_threshold {
        Mode::Cancel
    } else {
        Mode::Build
    };
    Plan { mode, opp_burst, opp_fill }
}
