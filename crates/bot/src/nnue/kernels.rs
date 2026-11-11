//! The three hot kernels of the forward pass (accumulator refresh, pairwise activation, u8 × i8 dot
//! product), behind one trait so the layer structure in `net.rs` is written once.
//!
//! Every kernel is generic over its (compile-time) length; `net.rs` dispatches on the network's width once
//! per call, so each supported width gets fully unrolled code. [`Scalar`] is plain fixed-size array loops
//! that LLVM vectorises for the build target. On x86-64, [`avx2`] has explicit AVX2 versions and the
//! AVX2-compiled entry points `net.rs` dispatches to at run time.

use super::net::FT_Q;

pub(super) trait Kernels {
    /// `acc = ft_b + Σ rows of ft_w` (rows are `L` long).
    fn refresh<const L: usize>(ft_b: &[i16; L], ft_w: &[i16], feats: &[u16], acc: &mut [i16; L]);
    /// `out[i] = clamp(a[i]) · clamp(b[i]) >> 9`.
    fn pairwise<const H: usize>(a: &[i16; H], b: &[i16; H], out: &mut [u8; H]);
    fn dot<const N: usize>(x: &[u8; N], w: &[i8; N]) -> i32;
}

pub(super) struct Scalar;

impl Kernels for Scalar {
    #[inline(always)]
    fn refresh<const L: usize>(ft_b: &[i16; L], ft_w: &[i16], feats: &[u16], acc: &mut [i16; L]) {
        *acc = *ft_b;
        for &f in feats {
            let i = f as usize * L;
            let row: &[i16; L] = ft_w[i..i + L].try_into().unwrap();
            for (a, w) in acc.iter_mut().zip(row) {
                *a = a.wrapping_add(*w);
            }
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

    use super::super::net::{self, Network, SideOut};
    use super::*;

    const TILE: usize = 128; // i16 lanes per register tile (8 registers)

    /// Only used by the `#[target_feature(enable = "avx2")]` entry points below, so its kernels always run
    /// with AVX2 available.
    struct Avx2;

    impl Kernels for Avx2 {
        #[inline(always)]
        fn refresh<const L: usize>(ft_b: &[i16; L], ft_w: &[i16], feats: &[u16], acc: &mut [i16; L]) {
            // SAFETY: see `Avx2`.
            unsafe { refresh(ft_b, ft_w, feats, acc) }
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
        net::side::<Avx2>(net, feats, out)
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn eval(net: &Network, me: &[u16], opp: &SideOut, bucket: usize) -> i32 {
        net::eval::<Avx2>(net, me, opp, bucket)
    }

    #[target_feature(enable = "avx2")]
    fn refresh<const L: usize>(ft_b: &[i16; L], ft_w: &[i16], feats: &[u16], acc: &mut [i16; L]) {
        const { assert!(L % TILE == 0) };
        for f in feats {
            assert!((*f as usize + 1) * L <= ft_w.len());
        }
        for t in 0..L / TILE {
            // SAFETY: every row index was bounds-checked above; unaligned loads/stores.
            unsafe {
                let bp = ft_b.as_ptr().add(t * TILE) as *const __m256i;
                let mut r: [__m256i; 8] = std::array::from_fn(|k| _mm256_loadu_si256(bp.add(k)));
                for &f in feats {
                    let wp = ft_w.as_ptr().add(f as usize * L + t * TILE) as *const __m256i;
                    for (k, reg) in r.iter_mut().enumerate() {
                        *reg = _mm256_add_epi16(*reg, _mm256_loadu_si256(wp.add(k)));
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
