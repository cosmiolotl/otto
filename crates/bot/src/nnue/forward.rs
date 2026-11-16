//! Integer forward passes, shared by portable and AVX2 kernels.

use super::kernels::Kernels;
use super::net::{A_Q, Accumulator, L2, L3, L3_IN, LayerStack, Network, SideOut};
use super::opponent::OpponentTable;

const W_SHIFT: u32 = 6;

/// Calls `$f::<$k, L1, L1 / 2>($args)` for the network's width (one monomorphised copy per width).
macro_rules! with_width {
    ($l1:expr, $f:ident::<$k:ty>($($arg:expr),*)) => {
        match $l1 {
            128 => $f::<$k, 128, 64>($($arg),*),
            256 => $f::<$k, 256, 128>($($arg),*),
            384 => $f::<$k, 384, 192>($($arg),*),
            512 => $f::<$k, 512, 256>($($arg),*),
            l1 => unreachable!("unsupported accumulator width {l1}"),
        }
    };
}

// The forward pass, once, over a kernel set. `#[inline(always)]` so that the AVX2 entry points
// (`kernels::avx2`) get their own copy compiled with AVX2 enabled.

#[inline(always)]
pub(super) fn side<K: Kernels>(net: &Network, feats: &[u16], out: &mut [u8]) {
    with_width!(net.l1, side_n::<K>(net, feats, out))
}

#[inline(always)]
fn side_n<K: Kernels, const L: usize, const H: usize>(net: &Network, feats: &[u16], out: &mut [u8]) {
    let mut acc = [0i16; L];
    K::refresh::<L>(net.ft_b.0[..L].try_into().unwrap(), &net.ft_w, feats, &mut acc);
    let (a, b) = acc.split_at(H);
    K::pairwise::<H>(a.try_into().unwrap(), b.try_into().unwrap(), (&mut out[..H]).try_into().unwrap());
}

#[inline(always)]
fn propagate<K: Kernels, const L: usize>(st: &LayerStack, input: &[u8; L]) -> i32 {
    let sums = std::array::from_fn(|o| st.l2_b[o] + K::dot::<L>(input, st.l2_w[o][..L].try_into().unwrap()));
    finish::<K>(st, &sums)
}

#[inline(always)]
fn finish<K: Kernels>(st: &LayerStack, sums: &[i32; L2]) -> i32 {
    let mut h2 = [0u8; L3_IN];
    for o in 0..L2 {
        let v = sums[o] >> W_SHIFT;
        let c = v.clamp(0, A_Q);
        h2[o] = c as u8;
        h2[L2 + o] = ((c * c) >> 7) as u8;
    }
    let mut h3 = [0u8; L3];
    for o in 0..L3 {
        let v = (st.l3_b[o] + K::dot(&h2, &st.l3_w[o])) >> W_SHIFT;
        h3[o] = v.clamp(0, A_Q) as u8;
    }
    st.out_b + K::dot(&h3, &st.out_w)
}

#[inline(always)]
pub(super) fn eval<K: Kernels>(net: &Network, me: &[u16], opp: &SideOut, bucket: usize) -> i32 {
    with_width!(net.l1, eval_n::<K>(net, me, opp, bucket))
}

#[inline(always)]
fn eval_n<K: Kernels, const L: usize, const H: usize>(net: &Network, me: &[u16], opp: &SideOut, bucket: usize) -> i32 {
    let mut input = [0u8; L];
    side_n::<K, L, H>(net, me, &mut input[..H]);
    input[H..].copy_from_slice(&opp[..H]);
    propagate::<K, L>(&net.stacks[bucket], &input)
}

#[inline(always)]
pub(super) fn eval_prepared<K: Kernels>(
    net: &Network,
    me: &[u16],
    opponent: &OpponentTable,
    pending: usize,
    bucket: usize,
) -> i32 {
    with_width!(net.l1, eval_prepared_n::<K>(net, me, opponent, pending, bucket))
}

#[inline(always)]
fn eval_prepared_n<K: Kernels, const L: usize, const H: usize>(
    net: &Network,
    me: &[u16],
    opponent: &OpponentTable,
    pending: usize,
    bucket: usize,
) -> i32 {
    let mut input = [0u8; H];
    side_n::<K, L, H>(net, me, &mut input);
    dense_prepared::<K, H>(net, &input, opponent, pending, bucket)
}

/// [`eval_prepared`] of the features of `src` plus `add` minus `sub`: `dst = src + Σ add rows − Σ sub rows`
/// (the first `l1` entries; bit-identical to a refresh), then the pairwise and dense layers from `dst`.
/// A full refresh is `src = net.ft_b` with every feature in `add`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn eval_update<K: Kernels>(
    net: &Network,
    src: &Accumulator,
    add: &[u16],
    sub: &[u16],
    dst: &mut Accumulator,
    opponent: &OpponentTable,
    pending: usize,
    bucket: usize,
) -> i32 {
    with_width!(net.l1, eval_update_n::<K>(net, src, add, sub, dst, opponent, pending, bucket))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn eval_update_n<K: Kernels, const L: usize, const H: usize>(
    net: &Network,
    src: &Accumulator,
    add: &[u16],
    sub: &[u16],
    dst: &mut Accumulator,
    opponent: &OpponentTable,
    pending: usize,
    bucket: usize,
) -> i32 {
    let acc: &mut [i16; L] = (&mut dst.0[..L]).try_into().unwrap();
    K::update::<L>((&src.0[..L]).try_into().unwrap(), &net.ft_w, add, sub, acc);
    let mut input = [0u8; H];
    let (a, b) = acc.split_at(H);
    K::pairwise::<H>(a.try_into().unwrap(), b.try_into().unwrap(), &mut input);
    dense_prepared::<K, H>(net, &input, opponent, pending, bucket)
}

/// `dst = src + Σ add rows − Σ sub rows` (the first `l1` entries).
#[inline(always)]
pub(super) fn update<K: Kernels>(net: &Network, src: &Accumulator, add: &[u16], sub: &[u16], dst: &mut Accumulator) {
    with_width!(net.l1, update_n::<K>(net, src, add, sub, dst))
}

#[inline(always)]
fn update_n<K: Kernels, const L: usize, const H: usize>(
    net: &Network,
    src: &Accumulator,
    add: &[u16],
    sub: &[u16],
    dst: &mut Accumulator,
) {
    K::update::<L>((&src.0[..L]).try_into().unwrap(), &net.ft_w, add, sub, (&mut dst.0[..L]).try_into().unwrap());
}

/// The dense layers from the side to move's pairwise output and the prepared opponent.
#[inline(always)]
fn dense_prepared<K: Kernels, const H: usize>(
    net: &Network,
    input: &[u8; H],
    opponent: &OpponentTable,
    pending: usize,
    bucket: usize,
) -> i32 {
    let stack = &net.stacks[bucket];
    let packed = &opponent.stacks[bucket];
    let sums = K::layer2(input, stack, packed, &opponent.l2[pending][bucket]);
    let h2 = K::activate_l2(&sums);
    let sums = K::layer3(&h2, stack, packed);
    let h3 = K::activate_l3(&sums);
    stack.out_b + K::dot(&h3, &stack.out_w)
}

#[inline(always)]
pub(super) fn prepare_opponent<K: Kernels>(net: &Network, feats: &[u16], table: &mut super::opponent::OpponentTable) {
    with_width!(net.l1, prepare_opponent_n::<K>(net, feats, table))
}

#[inline(always)]
fn prepare_opponent_n<K: Kernels, const L: usize, const H: usize>(
    net: &Network,
    feats: &[u16],
    table: &mut super::opponent::OpponentTable,
) {
    let mut base = [0i16; L];
    K::refresh::<L>(net.ft_b.0[..L].try_into().unwrap(), &net.ft_w, feats, &mut base);
    for (pending, side) in table.sides.iter_mut().enumerate() {
        // Only pending changes. Add its row to the shared, unclipped accumulator before activation.
        let feature = super::features::pending_feature(pending as u32);
        let mut acc = [0i16; L];
        K::refresh::<L>(&base, &net.ft_w, &[feature], &mut acc);
        let (a, b) = acc.split_at(H);
        K::pairwise::<H>(a.try_into().unwrap(), b.try_into().unwrap(), (&mut side[..H]).try_into().unwrap());
        for (bucket, stack) in net.stacks.iter().enumerate() {
            for o in 0..L2 {
                table.l2[pending][bucket][o] = stack.l2_b[o]
                    + K::dot::<H>((&side[..H]).try_into().unwrap(), stack.l2_w[o][H..L].try_into().unwrap());
            }
        }
    }
}
