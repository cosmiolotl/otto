//! NNUE-style value network (`docs/NNUE.md`).
//!
//! The network maps (side to move, opponent) to the logit of the probability that the side to move wins.
//! Its design follows Stockfish's NNUE: a sparse binary input layer whose weights are summed into int16
//! accumulators (one per player, shared weights), a pairwise-product activation, and a few tiny int8
//! dense layers selected by a bucket. The opponent's half is constant during a search and is prepared once
//! per move ([`Network::opponent_table`]), so a search node costs one accumulator refresh and the dense
//! layers.
//!
//! - [`features`]: the `rows-v1` feature set (one player's visible state → ≤ 89 active indices).
//! - [`net`]: quantized weights and the integer forward pass ([`kernels`]: scalar and AVX2).
//! - [`file`]: the `.nnue` file format.

// Kernels index several same-length arrays in lockstep; range loops read better there.
#![allow(clippy::needless_range_loop)]

pub mod features;
pub mod file;
mod kernels;
pub mod net;

pub use features::{Features, MAX_ACTIVE, N_BUCKETS, N_FEATURES, bucket, extract, layout_json};
pub use net::{Network, OpponentTable};
