//! A bot as rosters, matches and the tuner pass it around: a name, search options and an evaluator.

use std::sync::Arc;

use crate::bot::Bot;
use crate::eval::{AnyEval, HandEval, NnueEval};
use crate::nnue::Network;
use crate::search::SearchOptions;
use crate::weights::{self, Weights};

#[derive(Clone, Debug)]
pub enum EvalSpec {
    Hand(Weights),
    /// Equality is identity of the shared network.
    Nnue(Arc<Network>),
}

impl PartialEq for EvalSpec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (EvalSpec::Hand(a), EvalSpec::Hand(b)) => a == b,
            (EvalSpec::Nnue(a), EvalSpec::Nnue(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BotSpec {
    pub name: String,
    pub opts: SearchOptions,
    pub eval: EvalSpec,
}

impl BotSpec {
    pub fn hand(weights: Weights, opts: SearchOptions) -> BotSpec {
        BotSpec { name: weights.name.clone(), opts, eval: EvalSpec::Hand(weights) }
    }

    pub fn nnue(name: String, net: Arc<Network>, opts: SearchOptions) -> BotSpec {
        BotSpec { name, opts, eval: EvalSpec::Nnue(net) }
    }

    /// A `*.nnue` file is a network bot (named after the file); anything else is a weight preset or a
    /// weights file ([`weights::load`]).
    pub fn load(spec: &str, opts: SearchOptions) -> Result<BotSpec, String> {
        if spec.ends_with(".nnue") {
            let net = Network::load(spec).map_err(|e| e.to_string())?;
            Ok(BotSpec::nnue(weights::file_stem(spec), Arc::new(net), opts))
        } else {
            Ok(BotSpec::hand(weights::load(spec)?, opts))
        }
    }

    /// The hand-written weights, for a hand bot.
    pub fn weights(&self) -> Option<&Weights> {
        match &self.eval {
            EvalSpec::Hand(w) => Some(w),
            EvalSpec::Nnue(_) => None,
        }
    }

    pub fn make_bot(&self, seed: u64) -> Bot<AnyEval> {
        let eval = match &self.eval {
            EvalSpec::Hand(w) => AnyEval::Hand(Box::new(HandEval::new(w.clone()))),
            EvalSpec::Nnue(net) => AnyEval::Nnue(Box::new(NnueEval::new(net.clone()))),
        };
        Bot::with_eval(eval, self.opts, seed)
    }
}
