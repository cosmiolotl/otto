//! The quantized network and its integer forward pass (Stockfish-NNUE style).
//!
//! ```text
//!  me:  sparse features ──► FT (N_FEATURES → L1, i16, shared) ──► acc_me  ──► pairwise ──► 128 u8 ┐
//!  opp: sparse features ──► FT (same weights)                 ──► acc_opp ──► pairwise ──► 128 u8 ┴► 256 u8
//!       ──► bucket b = my stack height / 3 ──► L2[b] (256 → 16, i8) ──► [CReLU, SqrCReLU] 32 u8
//!       ──► L3[b] (32 → 32, i8) ──► CReLU ──► OUT[b] (32 → 1) ──► logit of P(side to move wins)
//! ```
//!
//! Quantization (float value = integer / scale):
//!
//! | Tensor | Type | Scale |
//! |---|---|---|
//! | FT weights, biases, accumulator | i16 | [`FT_Q`] = 255 (activation 1.0 = 255) |
//! | pairwise output `clamp(a)·clamp(b) >> 9` | u8 0..=127 | [`A_Q`] = 127 (≈ 255²/512) |
//! | dense weights | i8 | [`W_Q`] = 64 (so \|w\| ≤ 127/64) |
//! | dense biases, pre-activations | i32 | `A_Q · W_Q`; `>> 6` brings them back to `A_Q` |
//! | SqrCReLU | u8 | `clamp(v, 0, 127)² >> 7` |
//! | output | i32 | `A_Q · W_Q` (= 8128 per logit unit, [`Network::logit`]) |
//!
//! The accumulator adds i16 columns with wrapping arithmetic; the trainer clamps the (folded) FT weights
//! to ±1.0 and the biases to ±8.0 so that [`MAX_ACTIVE`](super::MAX_ACTIVE) active features can never
//! overflow it.

use tetris_sim::{PlayerState, SplitMix};

use super::features::{self, Features, N_BUCKETS, N_FEATURES, PENDING_MAX};
use super::kernels::{Kernels, Scalar};

/// Accumulator width per perspective.
pub const L1: usize = 256;
/// Pairwise-product outputs per perspective.
pub const HALF: usize = L1 / 2;
/// Dense input: both perspectives.
pub const L2_IN: usize = 2 * HALF;
pub const L2: usize = 16;
pub const L3_IN: usize = 2 * L2;
pub const L3: usize = 32;

pub const FT_Q: i32 = 255;
pub const A_Q: i32 = 127;
pub const W_Q: i32 = 64;
const W_SHIFT: u32 = 6;

/// One perspective's first-layer sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Accumulator(pub [i16; L1]);

/// Pairwise output of one perspective.
pub type SideOut = [u8; HALF];

/// The per-bucket dense layers.
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct LayerStack {
    pub l2_w: [[i8; L2_IN]; L2],
    pub l2_b: [i32; L2],
    pub l3_w: [[i8; L3_IN]; L3],
    pub l3_b: [i32; L3],
    pub out_w: [i8; L3],
    pub out_b: i32,
}

impl Default for LayerStack {
    fn default() -> Self {
        LayerStack {
            l2_w: [[0; L2_IN]; L2],
            l2_b: [0; L2],
            l3_w: [[0; L3_IN]; L3],
            l3_b: [0; L3],
            out_w: [0; L3],
            out_b: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Network {
    /// Free-form description from the file (trainer metadata).
    pub desc: String,
    pub ft_b: Box<Accumulator>,
    /// `N_FEATURES` rows of `L1` weights.
    pub ft_w: Vec<i16>,
    pub stacks: Vec<LayerStack>,
    /// Use the AVX2 kernels (runtime-detected; always false off x86-64).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    avx2: bool,
}

/// The opponent's perspective for each `pending` value (see [`Network::opponent_table`]).
#[derive(Clone, Debug)]
pub struct OpponentTable {
    pub base_pending: u32,
    pub sides: Box<[SideOut; PENDING_MAX + 1]>,
}

// The forward pass, once, over a kernel set. `#[inline(always)]` so that the AVX2 entry points
// (`kernels::avx2`) get their own copy compiled with AVX2 enabled.

#[inline(always)]
pub(super) fn side<K: Kernels>(net: &Network, feats: &[u16], out: &mut SideOut) {
    let mut acc = Accumulator([0; L1]);
    K::refresh(&net.ft_b, &net.ft_w, feats, &mut acc);
    K::pairwise(&acc, out);
}

#[inline(always)]
fn propagate<K: Kernels>(st: &LayerStack, input: &[u8; L2_IN]) -> i32 {
    let mut h2 = [0u8; L3_IN];
    for o in 0..L2 {
        let v = (st.l2_b[o] + K::dot(input, &st.l2_w[o])) >> W_SHIFT;
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
    let mut input = [0u8; L2_IN];
    let (a, b) = input.split_at_mut(HALF);
    side::<K>(net, me, a.try_into().unwrap());
    b.copy_from_slice(opp);
    propagate::<K>(&net.stacks[bucket], &input)
}

impl Network {
    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// A zero network (every output 0).
    pub fn zeroed() -> Network {
        Network {
            desc: String::new(),
            ft_b: Box::new(Accumulator([0; L1])),
            ft_w: vec![0; N_FEATURES * L1],
            stacks: vec![LayerStack::default(); N_BUCKETS],
            avx2: Self::detect_avx2(),
        }
    }

    /// Random weights in the trainer's clamp ranges (tests and benchmarks).
    pub fn random(seed: u64) -> Network {
        let mut rng = SplitMix(seed ^ 0x5851_f42d_4c95_7f2d);
        let mut r = |lo: i32, hi: i32| lo + rng.below((hi - lo + 1) as u64) as i32;
        let mut n = Network::zeroed();
        n.desc = format!("random seed {seed}");
        for b in n.ft_b.0.iter_mut() {
            *b = r(0, 2 * FT_Q / 3) as i16;
        }
        for w in n.ft_w.iter_mut() {
            *w = r(-FT_Q / 8, FT_Q / 8) as i16;
        }
        for s in n.stacks.iter_mut() {
            s.l2_w.iter_mut().flatten().for_each(|w| *w = r(-24, 24) as i8);
            s.l2_b.iter_mut().for_each(|b| *b = r(-A_Q * W_Q / 4, A_Q * W_Q / 4));
            s.l3_w.iter_mut().flatten().for_each(|w| *w = r(-48, 48) as i8);
            s.l3_b.iter_mut().for_each(|b| *b = r(-A_Q * W_Q / 4, A_Q * W_Q / 4));
            s.out_w.iter_mut().for_each(|w| *w = r(-64, 64) as i8);
            s.out_b = r(-A_Q * W_Q, A_Q * W_Q);
        }
        n
    }

    /// Integer output → logit (natural-log odds that the side to move wins).
    #[inline]
    pub fn logit(raw: i32) -> f32 {
        raw as f32 / (A_Q * W_Q) as f32
    }

    /// Refresh + pairwise for one perspective.
    pub fn side(&self, feats: &[u16], out: &mut SideOut) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 support was detected at construction.
            return unsafe { super::kernels::avx2::side(self, feats, out) };
        }
        side::<Scalar>(self, feats, out)
    }

    /// Raw output for the side to move (`me`) against a precomputed opponent perspective.
    pub fn eval(&self, me: &[u16], opp: &SideOut, bucket: usize) -> i32 {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 support was detected at construction.
            return unsafe { super::kernels::avx2::eval(self, me, opp, bucket) };
        }
        eval::<Scalar>(self, me, opp, bucket)
    }

    /// Raw output from two feature lists ([`Network::eval`] reuses a prepared opponent side).
    pub fn eval_pair(&self, me: &[u16], opp: &[u16], bucket: usize) -> i32 {
        let mut o = [0u8; HALF];
        self.side(opp, &mut o);
        self.eval(me, &o, bucket)
    }

    /// The opponent's perspective for every possible `pending` value: their board, pieces, chain and clock
    /// are fixed during a search, only the incoming total grows with what the searching player sends.
    pub fn opponent_table(&self, opp: &PlayerState, opp_fpp: u32) -> OpponentTable {
        let mut f = Features::default();
        features::extract_static(opp, opp_fpp, &mut f);
        let base_pending = opp.garbage.size();
        let n = f.len;
        let mut out = OpponentTable { base_pending, sides: Box::new([[0; HALF]; PENDING_MAX + 1]) };
        for (k, side) in out.sides.iter_mut().enumerate() {
            f.idx[n] = features::pending_feature(k as u32);
            f.len = n + 1;
            self.side(f.as_slice(), side);
        }
        out
    }

    /// Value of `me` (the side to move, `my_fpp` frames per piece) against a prepared opponent who has
    /// `sent` more incoming lines than at preparation time. Returns the logit.
    pub fn evaluate(&self, me: &PlayerState, my_fpp: u32, opp: &OpponentTable, sent: u32) -> f32 {
        let mut f = Features::default();
        features::extract(me, my_fpp, &mut f);
        let k = (opp.base_pending + sent).min(PENDING_MAX as u32) as usize;
        Network::logit(self.eval(f.as_slice(), &opp.sides[k], features::bucket(me)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::{Board, IncomingGarbage};

    fn sample_states() -> Vec<PlayerState> {
        let mut v = Vec::new();
        for seed in 1..6 {
            let mut s = PlayerState::tl(seed);
            s.board = Board::from_ascii(
                "
                ...#......
                .#.##.....
                ##.####.##
                ####.#####
                ",
            );
            s.chain.b2b = seed as i32 * 2 - 1;
            s.frame = seed as u32 * 1234;
            s.receive_garbage(IncomingGarbage { frame: 0, amount: seed as u32, cid: 1, gameid: 2 });
            v.push(s);
        }
        v.push(PlayerState::tl(99));
        v
    }

    /// Straightforward scalar forward pass in i32, written independently of the kernels.
    fn reference(net: &Network, me: &[u16], opp: &[u16], bucket: usize) -> i32 {
        let side = |feats: &[u16]| -> Vec<i32> {
            let mut acc: Vec<i32> = net.ft_b.0.iter().map(|&b| b as i32).collect();
            for &f in feats {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a += net.ft_w[f as usize * L1 + j] as i32;
                }
            }
            (0..HALF).map(|i| (acc[i].clamp(0, FT_Q) * acc[i + HALF].clamp(0, FT_Q)) >> 9).collect()
        };
        let input: Vec<i32> = side(me).into_iter().chain(side(opp)).collect();
        let st = &net.stacks[bucket];
        let mut h2 = [0i32; L3_IN];
        for o in 0..L2 {
            let v = (st.l2_b[o] + (0..L2_IN).map(|i| input[i] * st.l2_w[o][i] as i32).sum::<i32>()) >> 6;
            let c = v.clamp(0, 127);
            h2[o] = c;
            h2[L2 + o] = (c * c) >> 7;
        }
        let h3: Vec<i32> = (0..L3)
            .map(|o| ((st.l3_b[o] + (0..L3_IN).map(|i| h2[i] * st.l3_w[o][i] as i32).sum::<i32>()) >> 6).clamp(0, 127))
            .collect();
        st.out_b + (0..L3).map(|i| h3[i] * st.out_w[i] as i32).sum::<i32>()
    }

    #[test]
    fn optimised_matches_reference() {
        let net = Network::random(3);
        let states = sample_states();
        for (i, me) in states.iter().enumerate() {
            let opp = &states[(i + 1) % states.len()];
            let (mut a, mut b) = (Features::default(), Features::default());
            features::extract(me, 24, &mut a);
            features::extract(opp, 26, &mut b);
            let bk = features::bucket(me);
            let r = reference(&net, a.as_slice(), b.as_slice(), bk);
            assert_eq!(net.eval_pair(a.as_slice(), b.as_slice(), bk), r);
            let mut slow = net.clone();
            slow.avx2 = false;
            assert_eq!(slow.eval_pair(a.as_slice(), b.as_slice(), bk), r);
            // The opponent table gives the same as an opponent with `sent` more pending lines (that are not
            // ready to enter yet: a search path's attack only grows the pending total).
            let t = net.opponent_table(opp, 26);
            for sent in [0, 3, 50] {
                let mut hit = opp.clone();
                if sent > 0 {
                    let far = hit.frame + 1000;
                    hit.receive_garbage(IncomingGarbage { frame: far, amount: sent, cid: 9, gameid: 1 });
                }
                let mut c = Features::default();
                features::extract(&hit, 26, &mut c);
                let want = Network::logit(net.eval_pair(a.as_slice(), c.as_slice(), bk));
                assert_eq!(net.evaluate(me, 24, &t, sent), want);
            }
        }
    }
}
