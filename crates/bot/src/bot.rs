//! The bot: observation → evaluator plan (release controller) → beam search → placement.

use std::time::Instant;

use tetris_sim::{PieceQueue, Placement, SplitMix};

use crate::controller::Mode;
use crate::eval::{Evaluator, HandEval};
use crate::obs::Observation;
use crate::search::{Search, SearchOptions};
use crate::weights::Weights;

/// One move's decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub placement: Placement,
    pub mode: Mode,
    pub score: f32,
    pub nodes: u64,
}

/// Counters over the bot's lifetime (one game in self-play).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BotStats {
    pub moves: u64,
    /// Moves per controller mode ([`Mode::index`]).
    pub modes: [u64; Mode::COUNT],
    pub think_ns: u64,
}

pub struct Bot<E: Evaluator = HandEval> {
    pub opts: SearchOptions,
    pub eval: E,
    pub stats: BotStats,
    rng: SplitMix,
    search: Search,
    tails: Vec<PieceQueue>,
}

impl Bot<HandEval> {
    pub fn new(weights: Weights, opts: SearchOptions, seed: u64) -> Bot<HandEval> {
        Bot::with_eval(HandEval::new(weights), opts, seed)
    }
}

impl<E: Evaluator> Bot<E> {
    /// `seed` drives the bot's own randomness: the sampled queue and garbage continuations of the search.
    pub fn with_eval(eval: E, opts: SearchOptions, seed: u64) -> Bot<E> {
        Bot {
            opts,
            eval,
            stats: BotStats::default(),
            rng: SplitMix(seed),
            search: Search::default(),
            tails: Vec::new(),
        }
    }

    /// Choose a placement. `None` only if the current piece has no placement at all.
    pub fn choose(&mut self, obs: &Observation) -> Option<Decision> {
        let t0 = Instant::now();
        let root = obs.me.hypothetical(&mut self.rng);
        let plan = self.eval.begin(obs);
        self.tails.clear();
        if self.opts.extra_plies > 0 {
            for _ in 1..self.opts.continuations.max(1) {
                self.tails.push(obs.me.hypothetical(&mut self.rng).queue);
            }
        }
        let r = self.search.run(&root, obs.my_fpp, &self.eval, &self.opts, &self.tails)?;
        self.stats.moves += 1;
        self.stats.modes[plan.mode.index()] += 1;
        self.stats.think_ns += t0.elapsed().as_nanos() as u64;
        Some(Decision { placement: r.placement, mode: plan.mode, score: r.score, nodes: r.nodes })
    }
}
