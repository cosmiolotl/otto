//! The value network as a search evaluator.
//!
//! [`NnueEval::begin`] prepares the opponent's half of the network once per move from the observation (the
//! opponent is a static board during the search, `docs/BOT.md`). Each node is then valued as
//! `logit P(win)` of (node state, opponent + the lines this path sent them). There is no discounted path
//! term: everything the net needs is in the two states.
//!
//! **Controller.** With controller weights ([`NnueEval::with_controller`]), `begin` runs the release
//! controller exactly as [`HandEval`](super::HandEval) does with those weights (`controller::decide` if
//! `weights.controller`, else the default plan) and returns its plan, so the bot's mode statistics are
//! recorded. A `rows-v1-plan` network also reads the plan: its features are fixed for the move, so `begin`
//! computes them once and every node appends them to the side to move's features. A `rows-v1` network
//! ignores the plan (the controller then only labels the moves). Without controller weights the plan is the
//! default one (BUILD, no threat), which a `rows-v1-plan` network then sees.
//!
//! **Incremental accumulators.** The search hands over each parent's children together
//! ([`Evaluator::eval_children`]; other callers can announce the parent with [`Evaluator::expand`] and then
//! `eval` its children). A child's accumulator is computed from the parent's (as its children see
//! it) or from the previous child's when that is cheaper than a refresh, one child ahead with its new FT
//! rows prefetched ([`Network::evaluate_children`], bit-identical to a refresh; single nodes go through
//! [`Network::evaluate_cached`]). The references sit behind a `Mutex` taken with `try_lock`, so the
//! evaluator stays `Sync`: an evaluation that finds it busy (another thread) refreshes instead.

use std::sync::{Arc, Mutex, Weak};

use tetris_sim::{PlayerState, SplitMix};

use super::{Evaluator, PathAcc};
use crate::controller::burst::Burst;
use crate::controller::{self, Plan};
use crate::nnue::features::PLAN_ACTIVE;
use crate::nnue::{AccCache, Network, OpponentTable, plan_features};
use crate::obs::Observation;
use crate::weights::Weights;

#[derive(Debug)]
pub struct NnueEval {
    pub net: Arc<Network>,
    /// Release-controller weights (thresholds and `opp_burst_scale`); `None`: no controller.
    pub controller: Option<Weights>,
    my_fpp: u32,
    opp: Option<OpponentTable>,
    /// The move's plan features, and how many of them the network reads (0 for `rows-v1`).
    plan: [u16; PLAN_ACTIVE],
    plan_len: usize,
    burst: Burst,
    cache: Mutex<Cache>,
}

/// The last evaluated node and the network it was evaluated with (`Weak`, so that a replaced network's
/// address cannot be reused while the cache refers to it).
#[derive(Debug, Default)]
struct Cache {
    net: Weak<Network>,
    acc: AccCache,
}

impl Cache {
    /// The cache, emptied first if it belongs to another network.
    #[inline]
    fn of(&mut self, net: &Arc<Network>) -> &mut AccCache {
        if self.net.as_ptr() != Arc::as_ptr(net) {
            self.net = Arc::downgrade(net);
            self.acc.clear();
        }
        &mut self.acc
    }
}

impl Clone for NnueEval {
    fn clone(&self) -> Self {
        NnueEval {
            net: self.net.clone(),
            controller: self.controller.clone(),
            my_fpp: self.my_fpp,
            opp: self.opp.clone(),
            plan: self.plan,
            plan_len: self.plan_len,
            burst: self.burst.clone(),
            cache: Mutex::default(),
        }
    }
}

impl NnueEval {
    pub fn new(net: Arc<Network>) -> NnueEval {
        NnueEval::with_controller(net, None)
    }

    pub fn with_controller(net: Arc<Network>, controller: Option<Weights>) -> NnueEval {
        NnueEval {
            net,
            controller,
            my_fpp: 24,
            opp: None,
            plan: [0; PLAN_ACTIVE],
            plan_len: 0,
            burst: Burst::default(),
            cache: Mutex::default(),
        }
    }
}

impl Evaluator for NnueEval {
    fn begin(&mut self, obs: &Observation) -> Plan {
        // The opponent's features read only visible information, so any hypothetical completion works.
        let opp = obs.opp.hypothetical(&mut SplitMix(0));
        self.my_fpp = obs.my_fpp;
        self.opp = Some(self.net.opponent_table(&opp, obs.opp_fpp));
        let plan = match &self.controller {
            Some(w) if w.controller => controller::decide(obs, w, &mut self.burst),
            _ => Plan::default(),
        };
        if self.net.layout.has_plan() {
            self.plan = plan_features(&plan);
            self.plan_len = PLAN_ACTIVE;
        } else {
            self.plan_len = 0;
        }
        plan
    }

    #[inline]
    fn eval(&self, s: &PlayerState, path: &PathAcc) -> f32 {
        let opp = self.opp.as_ref().expect("NnueEval::begin was not called");
        let plan = &self.plan[..self.plan_len];
        let Ok(mut cache) = self.cache.try_lock() else {
            return self.net.evaluate_with(s, self.my_fpp, opp, path.sent_lines, plan);
        };
        self.net.evaluate_cached(cache.of(&self.net), s, self.my_fpp, opp, path.sent_lines, plan)
    }

    /// Prepares the parent reference for the children's [`Evaluator::eval`] calls.
    fn expand(&self, parent: &PlayerState) {
        if let Ok(mut cache) = self.cache.try_lock() {
            self.net.prepare_children(cache.of(&self.net), parent, self.my_fpp, &self.plan[..self.plan_len]);
        }
    }

    fn eval_children(&self, parent: &PlayerState, children: &[PlayerState], paths: &[PathAcc], out: &mut [f32]) {
        let opp = self.opp.as_ref().expect("NnueEval::begin was not called");
        let plan = &self.plan[..self.plan_len];
        let Ok(mut cache) = self.cache.try_lock() else {
            for ((c, p), o) in children.iter().zip(paths).zip(out.iter_mut()) {
                *o = self.net.evaluate_with(c, self.my_fpp, opp, p.sent_lines, plan);
            }
            return;
        };
        let sent = |i: usize| paths[i].sent_lines;
        self.net.evaluate_children(cache.of(&self.net), parent, children, sent, self.my_fpp, opp, plan, out);
    }
}
