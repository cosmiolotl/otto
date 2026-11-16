//! NNUE-style value network (`docs/NNUE.md`).
//!
//! The network maps (side to move, opponent) to the logit of the probability that the side to move wins.
//! Its design follows Stockfish's NNUE: a sparse binary input layer whose weights are summed into int16
//! accumulators (one per player, shared weights), a pairwise-product activation, and a few tiny int8
//! dense layers selected by a bucket. The opponent's half is constant during a search: its accumulator
//! and unshifted L2 contributions are prepared once per move ([`Network::opponent_table`]). A node refreshes
//! only its own accumulator and computes only its half of L2. Packed dense weights let SIMD lanes compute
//! separate outputs, avoiding a horizontal reduction for every neuron. All rounding and clipping stay
//! after the complete integer sums, so existing network files produce identical scores.
//!
//! - [`features`]: the `rows-v1` feature set (one player's visible state → ≤ 89 active indices), and
//!   `rows-v1-plan` (the same plus the side to move's controller plan, 3 more) ([`Layout`]).
//! - [`net`]: quantized weights; [`forward`] and [`kernels`]: portable and AVX2 forward passes.
//! - [`opponent`] and [`packed`]: per-search preparation (about 53 KiB beyond the opponent activations).
//! - [`incremental`]: the next node's accumulator from the last one's ([`Network::evaluate_cached`]).
//! - [`file`]: the `.nnue` file format.

// Kernels index several same-length arrays in lockstep; range loops read better there.
#![allow(clippy::needless_range_loop)]

pub mod features;
pub mod file;
mod forward;
pub mod incremental;
mod kernels;
pub mod net;
mod opponent;
mod packed;

pub use features::{
    Features, Layout, MAX_ACTIVE, MAX_ACTIVE_PLAN, N_BUCKETS, N_FEATURES, N_FEATURES_PLAN, bucket, extract,
    Slots, extract_slots, extract_with_plan, layout_json, plan_features, push_plan,
};
pub use incremental::AccCache;
pub use net::{Network, OpponentTable};
