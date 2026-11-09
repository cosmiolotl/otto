//! The value network as a search evaluator.
//!
//! [`NnueEval::begin`] prepares the opponent's half of the network once per move from the observation (the
//! opponent is a static board during the search, `docs/BOT.md`). Each node is then valued as
//! `logit P(win)` of (node state, opponent + the lines this path sent them). There is no controller and no
//! discounted path term: everything the net needs is in the two states.

use std::sync::Arc;

use tetris_sim::{PlayerState, SplitMix};

use super::{Evaluator, PathAcc};
use crate::controller::Plan;
use crate::nnue::{Network, OpponentTable};
use crate::obs::Observation;

#[derive(Clone, Debug)]
pub struct NnueEval {
    pub net: Arc<Network>,
    my_fpp: u32,
    opp: Option<OpponentTable>,
}

impl NnueEval {
    pub fn new(net: Arc<Network>) -> NnueEval {
        NnueEval { net, my_fpp: 24, opp: None }
    }
}

impl Evaluator for NnueEval {
    fn begin(&mut self, obs: &Observation) -> Plan {
        // The opponent's features read only visible information, so any hypothetical completion works.
        let opp = obs.opp.hypothetical(&mut SplitMix(0));
        self.my_fpp = obs.my_fpp;
        self.opp = Some(self.net.opponent_table(&opp, obs.opp_fpp));
        Plan::default()
    }

    #[inline]
    fn eval(&self, s: &PlayerState, path: &PathAcc) -> f32 {
        let opp = self.opp.as_ref().expect("NnueEval::begin was not called");
        self.net.evaluate(s, self.my_fpp, opp, path.sent_lines)
    }
}
