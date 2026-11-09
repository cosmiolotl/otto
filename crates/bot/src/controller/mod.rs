//! Release controller: picks KILL / CANCEL / BUILD once per move, before the search (PLAN.md M4).
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

/// Root-level threat estimates (all in lines).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Threat {
    /// `my_burst(k)`: most attack I can generate in the next k placements of the known queue.
    pub my_burst: f32,
    /// The same for the opponent (their known queue).
    pub opp_burst: f32,
    pub my_pending: f32,
    pub opp_pending: f32,
    /// Rows left before the visible top (20 - stack height).
    pub my_margin: f32,
    pub opp_margin: f32,
    /// `opp_margin + opp_burst_scale · opp_burst`.
    pub opp_absorb: f32,
    /// `my_burst - my_pending + opp_pending - opp_absorb` (> `kill_threshold` ⇒ KILL).
    pub kill_margin: f32,
    /// `sigmoid((my_pending + opp_burst_scale · opp_burst - my_margin) / 3)`.
    pub danger: f32,
    /// How full the opponent is: `clamp((opp_height + opp_pending - 8) / 12, 0, 1)` (scales the value of
    /// attack by `1 + pressure · opp_fill`).
    pub opp_fill: f32,
}

/// What the controller decided for one move.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Plan {
    pub mode: Mode,
    pub threat: Threat,
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
    let threat = Threat {
        my_burst,
        opp_burst,
        my_pending,
        opp_pending,
        my_margin,
        opp_margin,
        opp_absorb,
        kill_margin,
        danger,
        opp_fill,
    };
    let mode = if kill_margin > w.kill_threshold {
        Mode::Kill
    } else if my_pending > 0.0 && danger > w.cancel_threshold {
        Mode::Cancel
    } else {
        Mode::Build
    };
    Plan { mode, threat }
}
