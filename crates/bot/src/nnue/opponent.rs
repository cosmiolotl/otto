//! Opponent features and their first dense-layer contribution, prepared once per search.

use tetris_sim::PlayerState;

use super::features::{self, Features, N_BUCKETS, PENDING_MAX};
use super::forward;
use super::kernels::Scalar;
use super::net::{HALF_MAX, L2, Network, SideOut};
use super::packed::PackedStack;

/// The opponent's perspective for every possible pending-garbage value, tied to the preparing network.
/// Rebuild after changing network weights; `sides` exposes the prepared activations for inspection.
#[derive(Clone, Debug)]
pub struct OpponentTable {
    pub base_pending: u32,
    pub sides: Box<[SideOut; PENDING_MAX + 1]>,
    /// Bias plus the opponent's unshifted L2 dot product, indexed by pending, bucket, output.
    pub(super) l2: Box<[[[i32; L2]; N_BUCKETS]; PENDING_MAX + 1]>,
    pub(super) stacks: Vec<PackedStack>,
}

impl Network {
    /// Prepare the fixed opponent state for a search. Only pending garbage changes along its paths.
    pub fn opponent_table(&self, opp: &PlayerState, opp_fpp: u32) -> OpponentTable {
        let mut features = Features::default();
        features::extract_static(opp, opp_fpp, &mut features);
        let mut table = OpponentTable {
            base_pending: opp.garbage.size(),
            sides: Box::new([[0; HALF_MAX]; PENDING_MAX + 1]),
            l2: Box::new([[[0; L2]; N_BUCKETS]; PENDING_MAX + 1]),
            stacks: self.stacks.iter().map(|stack| PackedStack::new(stack, self.l1 / 2)).collect(),
        };
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 support was detected at construction.
            unsafe { super::kernels::avx2::prepare_opponent(self, features.as_slice(), &mut table) };
            return table;
        }
        forward::prepare_opponent::<Scalar>(self, features.as_slice(), &mut table);
        table
    }

    pub(super) fn eval_prepared(&self, me: &[u16], opponent: &OpponentTable, pending: usize, bucket: usize) -> i32 {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 support was detected at construction.
            return unsafe { super::kernels::avx2::eval_prepared(self, me, opponent, pending, bucket) };
        }
        forward::eval_prepared::<Scalar>(self, me, opponent, pending, bucket)
    }
}
