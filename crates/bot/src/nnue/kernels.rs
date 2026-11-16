//! Hot kernels: accumulator update (refresh = update from the biases), pairwise activation, the dense
//! layers and their activations. One trait shares the layer structure in `forward.rs` between portable and
//! AVX2 implementations.
//!
//! Every kernel is generic over its (compile-time) length; `forward.rs` dispatches on the network's width once
//! per call, so each supported width gets fully unrolled code. [`Scalar`] is plain fixed-size array loops
//! that LLVM vectorises for the build target (aarch64 runs these): accumulator tiles held in registers, and
//! dense layers as dot products over the unpacked weights. On x86-64, [`avx2`] has explicit AVX2 versions
//! (packed dense weights, an L2 that skips zero input groups, the slot diff of `incremental`) and the
//! AVX2-compiled entry points selected at run time. All of them compute the same integers.

use super::net::{A_Q, FT_Q, L2, L3, L3_IN, LayerStack};
use super::packed::PackedStack;

pub(super) trait Kernels {
    /// `acc = ft_b + Σ rows of ft_w` (rows are `L` long).
    #[inline(always)]
    fn refresh<const L: usize>(ft_b: &[i16; L], ft_w: &[i16], feats: &[u16], acc: &mut [i16; L]) {
        Self::update(ft_b, ft_w, feats, &[], acc)
    }
    /// `acc = src + Σ add rows − Σ sub rows`, wrapping like [`Kernels::refresh`] (so an accumulator
    /// updated from another one is bit-identical to a refresh of its features).
    fn update<const L: usize>(src: &[i16; L], ft_w: &[i16], add: &[u16], sub: &[u16], acc: &mut [i16; L]);
    /// `out[i] = clamp(a[i]) · clamp(b[i]) >> 9`.
    fn pairwise<const H: usize>(a: &[i16; H], b: &[i16; H], out: &mut [u8; H]);
    fn dot<const N: usize>(x: &[u8; N], w: &[i8; N]) -> i32;
    /// The side to move's half of L2 (`bias` holds the rest): one dot product per output over the unpacked
    /// weights. That is what the portable kernels vectorise best; [`avx2`] uses the packed weights.
    #[inline(always)]
    fn layer2<const H: usize>(
        input: &[u8; H],
        stack: &LayerStack,
        packed: &PackedStack,
        bias: &[i32; L2],
    ) -> [i32; L2] {
        let _ = packed;
        std::array::from_fn(|o| bias[o] + Self::dot::<H>(input, stack.l2_w[o][..H].try_into().unwrap()))
    }
    /// L3, as [`Kernels::layer2`].
    #[inline(always)]
    fn layer3(h2: &[u8; L3_IN], stack: &LayerStack, packed: &PackedStack) -> [i32; L3] {
        let _ = packed;
        std::array::from_fn(|o| stack.l3_b[o] + Self::dot(h2, &stack.l3_w[o]))
    }
    /// The L2 activation: `c = clamp(sum >> 6, 0, 127)`, then `[c ‖ c² >> 7]`.
    #[inline(always)]
    fn activate_l2(sums: &[i32; L2]) -> [u8; L3_IN] {
        let mut h2 = [0u8; L3_IN];
        for o in 0..L2 {
            let c = (sums[o] >> 6).clamp(0, A_Q);
            h2[o] = c as u8;
            h2[L2 + o] = ((c * c) >> 7) as u8;
        }
        h2
    }
    /// The L3 activation: `clamp(sum >> 6, 0, 127)`.
    #[inline(always)]
    fn activate_l3(sums: &[i32; L3]) -> [u8; L3] {
        let mut h3 = [0u8; L3];
        for o in 0..L3 {
            h3[o] = (sums[o] >> 6).clamp(0, A_Q) as u8;
        }
        h3
    }
}

pub(super) struct Scalar;

/// i16 lanes per accumulator tile of the portable kernels (every supported width is a multiple).
const SCALAR_TILE: usize = 64;

impl Kernels for Scalar {
    #[inline(always)]
    fn update<const L: usize>(src: &[i16; L], ft_w: &[i16], add: &[u16], sub: &[u16], acc: &mut [i16; L]) {
        const { assert!(L % SCALAR_TILE == 0) };
        // A tile of the accumulator stays in registers while the rows are added (8 registers of 8 lanes
        // on NEON or SSE2), instead of a load and a store of the accumulator for every row. The row bounds
        // are checked first: a check inside the row loop makes LLVM store the tile on every row.
        for &f in add.iter().chain(sub) {
            assert!((f as usize + 1) * L <= ft_w.len(), "feature {f} out of range");
        }
        let row = |f: u16, t: usize| {
            // SAFETY: row `f` is in bounds (checked above) and `t + SCALAR_TILE <= L`.
            unsafe { &*ft_w.as_ptr().add(f as usize * L + t).cast::<[i16; SCALAR_TILE]>() }
        };
        for t in (0..L).step_by(SCALAR_TILE) {
            let mut r: [i16; SCALAR_TILE] = src[t..t + SCALAR_TILE].try_into().unwrap();
            for &f in add {
                let w = row(f, t);
                for k in 0..SCALAR_TILE {
                    r[k] = r[k].wrapping_add(w[k]);
                }
            }
            for &f in sub {
                let w = row(f, t);
                for k in 0..SCALAR_TILE {
                    r[k] = r[k].wrapping_sub(w[k]);
                }
            }
            acc[t..t + SCALAR_TILE].copy_from_slice(&r);
        }
    }

    #[inline(always)]
    fn pairwise<const H: usize>(a: &[i16; H], b: &[i16; H], out: &mut [u8; H]) {
        for i in 0..H {
            let x = a[i].clamp(0, FT_Q as i16) as u32;
            let y = b[i].clamp(0, FT_Q as i16) as u32;
            out[i] = ((x * y) >> 9) as u8;
        }
    }

    #[inline(always)]
    fn dot<const N: usize>(x: &[u8; N], w: &[i8; N]) -> i32 {
        let mut s = 0i32;
        for i in 0..N {
            s += x[i] as i32 * w[i] as i32;
        }
        s
    }
}

/// Explicit AVX2 kernels. The accumulator is kept in registers while feature rows are added (tiles of
/// 8 × 16 lanes), the pairwise product uses `mullo` + shift + `packus`, and the dot products use
/// `maddubs` (u8 × i8 → i16 pairs; inputs ≤ 127 so the pair sums cannot saturate) + `madd`.
#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
    use std::arch::x86_64::*;

    use super::super::features::{SLOTS, Slots};
    use super::super::net::{Accumulator, Network, SideOut};
    use super::super::{forward, opponent};
    use super::*;

    const TILE: usize = 128; // i16 lanes per register tile (8 registers)

    /// Only used by the `#[target_feature(enable = "avx2")]` entry points below, so its kernels always run
    /// with AVX2 available.
    struct Avx2;

    impl Kernels for Avx2 {
        #[inline(always)]
        fn update<const L: usize>(src: &[i16; L], ft_w: &[i16], add: &[u16], sub: &[u16], acc: &mut [i16; L]) {
            // SAFETY: see `Avx2`.
            unsafe { update(src, ft_w, add, sub, acc) }
        }

        #[inline(always)]
        fn layer2<const H: usize>(
            input: &[u8; H],
            stack: &LayerStack,
            packed: &PackedStack,
            bias: &[i32; L2],
        ) -> [i32; L2] {
            let _ = stack;
            // SAFETY: see `Avx2`.
            unsafe { affine_sparse(input, &packed.l2[..H / 4], bias) }
        }

        #[inline(always)]
        fn layer3(h2: &[u8; L3_IN], stack: &LayerStack, packed: &PackedStack) -> [i32; L3] {
            // SAFETY: see `Avx2`.
            unsafe { affine(h2, &packed.l3, &stack.l3_b) }
        }

        #[inline(always)]
        fn activate_l2(sums: &[i32; L2]) -> [u8; L3_IN] {
            // SAFETY: see `Avx2`.
            unsafe { activate_l2(sums) }
        }

        #[inline(always)]
        fn activate_l3(sums: &[i32; L3]) -> [u8; L3] {
            // SAFETY: see `Avx2`.
            unsafe { activate_l3(sums) }
        }

        #[inline(always)]
        fn pairwise<const H: usize>(a: &[i16; H], b: &[i16; H], out: &mut [u8; H]) {
            // SAFETY: see `Avx2`.
            unsafe { pairwise(a, b, out) }
        }

        #[inline(always)]
        fn dot<const N: usize>(x: &[u8; N], w: &[i8; N]) -> i32 {
            // SAFETY: see `Avx2`.
            unsafe { dot(x, w) }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn side(net: &Network, feats: &[u16], out: &mut SideOut) {
        forward::side::<Avx2>(net, feats, out)
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn eval(net: &Network, me: &[u16], opp: &SideOut, bucket: usize) -> i32 {
        forward::eval::<Avx2>(net, me, opp, bucket)
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn eval_prepared(
        net: &Network,
        me: &[u16],
        opp: &opponent::OpponentTable,
        pending: usize,
        bucket: usize,
    ) -> i32 {
        forward::eval_prepared::<Avx2>(net, me, opp, pending, bucket)
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn update_acc(
        net: &Network,
        src: &Accumulator,
        add: &[u16],
        sub: &[u16],
        dst: &mut Accumulator,
    ) {
        forward::update::<Avx2>(net, src, add, sub, dst)
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(in super::super) fn eval_update(
        net: &Network,
        src: &Accumulator,
        add: &[u16],
        sub: &[u16],
        dst: &mut Accumulator,
        opp: &opponent::OpponentTable,
        pending: usize,
        bucket: usize,
    ) -> i32 {
        forward::eval_update::<Avx2>(net, src, add, sub, dst, opp, pending, bucket)
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn prepare_opponent(net: &Network, feats: &[u16], table: &mut opponent::OpponentTable) {
        forward::prepare_opponent::<Avx2>(net, feats, table)
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn affine<const I: usize, const O: usize, const B: usize>(
        input: &[u8; I],
        weights: &[[i8; B]],
        bias: &[i32; O],
    ) -> [i32; O] {
        const { assert!(I % 4 == 0 && B == O * 4 && O % 8 == 0 && O <= 32) };
        assert_eq!(weights.len(), I / 4);
        // SAFETY: fixed-sized rows hold 4 weights per output. Inputs are activations in 0..=127,
        // so maddubs' signed i16 pair sums cannot saturate, even for weights of -128.
        unsafe {
            let mut sums = [_mm256_setzero_si256(); 4];
            for o in 0..O / 8 {
                sums[o] = _mm256_loadu_si256(bias.as_ptr().add(o * 8).cast());
            }
            let ones = _mm256_set1_epi16(1);
            for (values, row) in input.chunks_exact(4).zip(weights) {
                let broadcast = _mm256_set1_epi32(i32::from_le_bytes(values.try_into().unwrap()));
                for o in 0..O / 8 {
                    let weights = _mm256_loadu_si256(row.as_ptr().add(o * 32).cast());
                    let pairs = _mm256_maddubs_epi16(broadcast, weights);
                    sums[o] = _mm256_add_epi32(sums[o], _mm256_madd_epi16(pairs, ones));
                }
            }
            let mut output = [0; O];
            for o in 0..O / 8 {
                _mm256_storeu_si256(output.as_mut_ptr().add(o * 8).cast(), sums[o]);
            }
            output
        }
    }

    /// [`affine`] for a sparse input (the pairwise output): only the non-zero groups of four inputs (found 32
    /// bytes at a time) are multiplied, which leaves the integer sums unchanged.
    #[target_feature(enable = "avx2")]
    pub(super) fn affine_sparse<const I: usize, const O: usize, const B: usize>(
        input: &[u8; I],
        weights: &[[i8; B]],
        bias: &[i32; O],
    ) -> [i32; O] {
        const { assert!(I % 32 == 0 && I / 4 <= 128 && B == O * 4 && O % 8 == 0 && O <= 32) };
        assert_eq!(weights.len(), I / 4);
        // SAFETY: as in `affine`; group indices come from the mask of `I / 4` groups.
        unsafe {
            let zero = _mm256_setzero_si256();
            // Bit g set: input group g (4 bytes) is not all zero. Inputs are 0..=127, so a group is
            // non-zero exactly when it is positive as an i32.
            let mut nnz = [0u32; 4];
            for c in 0..I / 32 {
                let v = _mm256_loadu_si256(input.as_ptr().add(32 * c).cast());
                let m = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero))) as u32;
                nnz[c / 4] |= m << (8 * (c % 4));
            }
            let mut sums = [_mm256_setzero_si256(); 4];
            for o in 0..O / 8 {
                sums[o] = _mm256_loadu_si256(bias.as_ptr().add(o * 8).cast());
            }
            let ones = _mm256_set1_epi16(1);
            for (word, &bits) in nnz.iter().enumerate().take(I.div_ceil(128)) {
                let mut m = bits;
                while m != 0 {
                    let g = 32 * word + m.trailing_zeros() as usize;
                    m &= m - 1;
                    let values = input.as_ptr().add(4 * g).cast::<i32>().read_unaligned();
                    let broadcast = _mm256_set1_epi32(values);
                    let row = weights.get_unchecked(g);
                    for o in 0..O / 8 {
                        let w = _mm256_loadu_si256(row.as_ptr().add(o * 32).cast());
                        let pairs = _mm256_maddubs_epi16(broadcast, w);
                        sums[o] = _mm256_add_epi32(sums[o], _mm256_madd_epi16(pairs, ones));
                    }
                }
            }
            let mut output = [0; O];
            for o in 0..O / 8 {
                _mm256_storeu_si256(output.as_mut_ptr().add(o * 8).cast(), sums[o]);
            }
            output
        }
    }

    /// `clamp(v >> 6, 0, 127)` for 8 lanes.
    #[target_feature(enable = "avx2")]
    fn clamp_shift(v: __m256i) -> __m256i {
        _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32::<6>(v), _mm256_setzero_si256()), _mm256_set1_epi32(A_Q))
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn activate_l2(sums: &[i32; L2]) -> [u8; L3_IN] {
        const { assert!(L2 == 16 && L3_IN == 32) };
        // SAFETY: fixed-size arrays.
        unsafe {
            let c0 = clamp_shift(_mm256_loadu_si256(sums.as_ptr().cast()));
            let c1 = clamp_shift(_mm256_loadu_si256(sums.as_ptr().add(8).cast()));
            let q0 = _mm256_srli_epi32::<7>(_mm256_mullo_epi32(c0, c0));
            let q1 = _mm256_srli_epi32::<7>(_mm256_mullo_epi32(c1, c1));
            // Values are 0..=127, so the packs do not saturate. Dwords after packing: c0[0..4], c1[0..4],
            // q0[0..4], q1[0..4], c0[4..8], c1[4..8], q0[4..8], q1[4..8].
            let bytes = _mm256_packus_epi16(_mm256_packs_epi32(c0, c1), _mm256_packs_epi32(q0, q1));
            let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let mut out = [0u8; L3_IN];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), _mm256_permutevar8x32_epi32(bytes, order));
            out
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn activate_l3(sums: &[i32; L3]) -> [u8; L3] {
        const { assert!(L3 == 32) };
        // SAFETY: fixed-size arrays.
        unsafe {
            let v: [__m256i; 4] =
                std::array::from_fn(|k| clamp_shift(_mm256_loadu_si256(sums.as_ptr().add(8 * k).cast())));
            // Dwords after packing: v0[0..4], v1[0..4], v2[0..4], v3[0..4], v0[4..8], v1[4..8], v2[4..8], v3[4..8].
            let bytes = _mm256_packus_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
            let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let mut out = [0u8; L3];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), _mm256_permutevar8x32_epi32(bytes, order));
            out
        }
    }

    #[target_feature(enable = "avx2")]
    fn update<const L: usize>(src: &[i16; L], ft_w: &[i16], add: &[u16], sub: &[u16], acc: &mut [i16; L]) {
        const { assert!(L % TILE == 0) };
        for f in add.iter().chain(sub) {
            assert!((*f as usize + 1) * L <= ft_w.len());
        }
        for t in 0..L / TILE {
            // SAFETY: every row index was bounds-checked above; unaligned loads/stores.
            unsafe {
                let bp = src.as_ptr().add(t * TILE) as *const __m256i;
                let mut r: [__m256i; 8] = std::array::from_fn(|k| _mm256_loadu_si256(bp.add(k)));
                for &f in add {
                    let wp = ft_w.as_ptr().add(f as usize * L + t * TILE) as *const __m256i;
                    for (k, reg) in r.iter_mut().enumerate() {
                        *reg = _mm256_add_epi16(*reg, _mm256_loadu_si256(wp.add(k)));
                    }
                }
                for &f in sub {
                    let wp = ft_w.as_ptr().add(f as usize * L + t * TILE) as *const __m256i;
                    for (k, reg) in r.iter_mut().enumerate() {
                        *reg = _mm256_sub_epi16(*reg, _mm256_loadu_si256(wp.add(k)));
                    }
                }
                let ap = acc.as_mut_ptr().add(t * TILE) as *mut __m256i;
                for (k, reg) in r.iter().enumerate() {
                    _mm256_storeu_si256(ap.add(k), *reg);
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    fn pairwise<const H: usize>(a: &[i16; H], b: &[i16; H], out: &mut [u8; H]) {
        const { assert!(H % 32 == 0) };
        // SAFETY: fixed-size arrays; H is a multiple of 32.
        unsafe {
            let zero = _mm256_setzero_si256();
            let max = _mm256_set1_epi16(FT_Q as i16);
            let (pa, pb) = (a.as_ptr() as *const __m256i, b.as_ptr() as *const __m256i);
            let q = out.as_mut_ptr() as *mut __m256i;
            let prod = |i: usize| {
                let x = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(pa.add(i)), zero), max);
                let y = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(pb.add(i)), zero), max);
                _mm256_srli_epi16::<9>(_mm256_mullo_epi16(x, y))
            };
            for j in 0..H / 32 {
                let packed = _mm256_packus_epi16(prod(2 * j), prod(2 * j + 1));
                _mm256_storeu_si256(q.add(j), _mm256_permute4x64_epi64::<0b11_01_10_00>(packed));
            }
        }
    }

    /// `LUT[m]`: the `pshufb` control that packs the u16 lanes `k` with bit `k` of `m` set to the front.
    static COMPRESS: [[u8; 16]; 256] = {
        let mut t = [[0x80u8; 16]; 256];
        let mut m = 0;
        while m < 256 {
            let (mut n, mut k) = (0, 0);
            while k < 8 {
                if (m >> k) & 1 == 1 {
                    t[m][2 * n] = 2 * k as u8;
                    t[m][2 * n + 1] = 2 * k as u8 + 1;
                    n += 1;
                }
                k += 1;
            }
            m += 1;
        }
        t
    };

    /// `incremental::diff_portable`, 8 slots at a time: compare, then pack the differing slots' features
    /// with `pshufb` (no branches: which slots differ is unpredictable).
    #[target_feature(enable = "avx2,popcnt")]
    pub(in super::super) fn diff(
        old: &Slots,
        new: &Slots,
        add: &mut [u16; SLOTS],
        sub: &mut [u16; SLOTS],
    ) -> (usize, usize) {
        const { assert!(SLOTS % 8 == 0) };
        let (mut na, mut ns) = (0usize, 0usize);
        // SAFETY: in-bounds loads of 8 slots. Before group g at most 8g slots were packed, so each 16-byte
        // store at `add + na` / `sub + ns` ends within the SLOTS-long buffers.
        unsafe {
            let none = _mm_set1_epi16(-1);
            let zero = _mm_setzero_si128();
            for g in 0..SLOTS / 8 {
                let a = _mm_loadu_si128(old.0.as_ptr().add(8 * g).cast());
                let b = _mm_loadu_si128(new.0.as_ptr().add(8 * g).cast());
                let eq = _mm_cmpeq_epi16(a, b);
                // Bit k set: lane k is kept (it differs and is not empty).
                let keep = |x: __m128i| {
                    let skip = _mm_or_si128(eq, _mm_cmpeq_epi16(x, none));
                    !_mm_movemask_epi8(_mm_packs_epi16(skip, zero)) as u32 & 0xff
                };
                let (ms, ma) = (keep(a), keep(b));
                let packed_sub = _mm_shuffle_epi8(a, _mm_loadu_si128(COMPRESS[ms as usize].as_ptr().cast()));
                let packed_add = _mm_shuffle_epi8(b, _mm_loadu_si128(COMPRESS[ma as usize].as_ptr().cast()));
                _mm_storeu_si128(sub.as_mut_ptr().add(ns).cast(), packed_sub);
                _mm_storeu_si128(add.as_mut_ptr().add(na).cast(), packed_add);
                ns += ms.count_ones() as usize;
                na += ma.count_ones() as usize;
            }
        }
        (na, ns)
    }

    #[target_feature(enable = "avx2")]
    fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "avx2")]
    fn dot<const N: usize>(x: &[u8; N], w: &[i8; N]) -> i32 {
        const { assert!(N % 32 == 0) };
        // SAFETY: N is a multiple of 32; unaligned loads.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_si256();
            for i in (0..N).step_by(32) {
                let a = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
                let b = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
            }
            hsum(acc)
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use tetris_sim::SplitMix;

    /// The AVX2 dense layers (packed weights, sparse L2 input) equal the portable ones (dot products over the
    /// unpacked weights) on sparse and dense inputs, extreme weights, and every half width.
    #[test]
    fn avx2_dense_kernels_match_portable() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt")) {
            return;
        }
        fn check<const H: usize>(input: &[u8; H], stack: &LayerStack, bias: &[i32; L2]) {
            let packed = PackedStack::new(stack, H);
            // SAFETY: AVX2 is available.
            let fast = unsafe { avx2::affine_sparse(input, &packed.l2[..H / 4], bias) };
            assert_eq!(fast, Scalar::layer2::<H>(input, stack, &packed, bias));
        }
        let mut rng = SplitMix(9);
        for round in 0..300 {
            let mut stack = LayerStack::default();
            for w in stack.l2_w.iter_mut().flatten().chain(stack.l3_w.iter_mut().flatten()) {
                *w = if round % 7 == 0 { -128 } else { rng.below(256) as u8 as i8 };
            }
            stack.l3_b = std::array::from_fn(|_| rng.below(20_000) as i32 - 10_000);
            let mut input = [0u8; 256];
            for v in input.iter_mut() {
                if rng.below(5) < round % 5 {
                    *v = rng.below(128) as u8;
                }
            }
            let bias: [i32; L2] = std::array::from_fn(|_| rng.below(20_000) as i32 - 10_000);
            check::<64>(input[..64].try_into().unwrap(), &stack, &bias);
            check::<128>(input[..128].try_into().unwrap(), &stack, &bias);
            check::<192>(input[..192].try_into().unwrap(), &stack, &bias);
            check::<256>(&input, &stack, &bias);
            let h2: [u8; L3_IN] = std::array::from_fn(|_| rng.below(128) as u8);
            let packed = PackedStack::new(&stack, 128);
            // SAFETY: AVX2 is available.
            let fast = unsafe { avx2::affine(&h2, &packed.l3, &stack.l3_b) };
            assert_eq!(fast, Scalar::layer3(&h2, &stack, &packed));
            let sums2: [i32; L2] = std::array::from_fn(|_| rng.below(40_000) as i32 - 15_000);
            let sums3: [i32; L3] = std::array::from_fn(|_| rng.below(40_000) as i32 - 15_000);
            // SAFETY: AVX2 is available.
            unsafe {
                assert_eq!(avx2::activate_l2(&sums2), Scalar::activate_l2(&sums2));
                assert_eq!(avx2::activate_l3(&sums3), Scalar::activate_l3(&sums3));
            }
        }
    }
}
