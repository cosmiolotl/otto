//! The Otto engine for TETR.IO S2 versus: beam search over placements ([`search`]), scored by the
//! hand-written evaluation over a flat weight vector with a burst/reserve release controller
//! ([`eval::HandEval`], [`controller`], [`weights`]) or by the NNUE value network ([`eval::NnueEval`],
//! [`nnue`]). See `docs/BOT.md` and `docs/NNUE.md`.
//!
//! ```no_run
//! use tetris_bot::{Bot, Observation, SearchOptions, weights};
//! use tetris_sim::Match;
//!
//! let mut m = Match::tl(12345, [24, 25]);
//! let mut bots = [
//!     Bot::new(weights::preset("default").unwrap(), SearchOptions::default(), 1),
//!     Bot::new(weights::preset("greedy-app").unwrap(), SearchOptions::default(), 2),
//! ];
//! while let Some(p) = m.next_turn() {
//!     let d = bots[p].choose(&Observation::from_match(&m, p)).unwrap();
//!     m.play(p, d.placement);
//! }
//! ```
//!
//! The bot only ever sees an [`Observation`] (no bag RNG, no garbage-hole RNG); its search runs on
//! hypothetical states built from it ([`PlayerView::hypothetical`]).

pub mod bot;
pub mod controller;
pub mod eval;
mod hash;
pub mod nnue;
pub mod obs;
pub mod search;
pub mod spec;
pub mod weights;

pub use bot::{Bot, BotStats, Decision};
pub use controller::{Mode, Plan};
pub use eval::{AnyEval, Evaluator, HandEval, NnueEval, PathAcc};
pub use hash::state_hash;
pub use obs::{Observation, PlayerView};
pub use search::{DEFAULT_WIDTH, GEN_MODE, Search, SearchOptions};
pub use spec::{BotSpec, EvalSpec};
pub use weights::Weights;
