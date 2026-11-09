//! Self-play: everything that plays many bot-vs-bot games (`docs/SELFPLAY.md`, `docs/NNUE.md`).
//!
//! - [`game`]: one deterministic game with its metrics (spikes, cause of loss), optionally recorded as an
//!   oracle request. Every other module plays games through it.
//! - [`conditions`]: what a pair of games is played under (speeds, rules, seeds), drawn per pair.
//! - [`schedule`]: paired-seed jobs (both seatings of every draw), run in parallel with rayon.
//! - [`report`]: per-bot totals (APP, spikes, cancelled/landed, controller modes, loss causes).
//! - [`stats`], [`rating`]: confidence intervals and tests; Bradley-Terry ratings with bootstrap CIs.
//! - [`headtohead`]: two bots at fixed speeds (`sim bot-vs-bot`).
//! - [`tournament`]: round robin over a roster (`sim tournament`).
//! - [`cmaes`], [`tune`]: CMA-ES over [`tetris_bot::Weights`] with fitness = score against a mixed pool,
//!   checkpointed every generation so a run can resume (`sim tune`).
//! - [`datagen`]: training data for the value network and its file format (`sim datagen`).

pub mod cmaes;
pub mod conditions;
pub mod datagen;
pub mod game;
pub mod headtohead;
pub mod rating;
pub mod report;
pub mod schedule;
pub mod stats;
pub mod tournament;
pub mod tune;

pub use conditions::{Conditions, FixedSpeed, PairDraw, fpp_of_pps};
pub use game::{GameResult, GameSpec, LossCause, SeatMetrics, play_game, play_game_with};
pub use rating::{Ratings, bradley_terry};
pub use schedule::{Job, PairResult};
