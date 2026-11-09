//! The three hot kernels of the forward pass (accumulator refresh, pairwise activation, u8 × i8 dot
//! product), behind one trait so the layer structure in `net.rs` is written once.
//!
//! [`Scalar`] is plain fixed-size array loops that LLVM vectorises for the build target. On x86-64,
//! [`avx2`] has explicit AVX2 versions and the AVX2-compiled entry points `net.rs` dispatches to at run
//! time.

use super::net::{Accumulator, FT_Q, HALF, L1, SideOut};

pub(super) trait Kernels {
    fn refresh(ft_b: &Accumulator, ft_w: &[i16], feats: &[u16], acc: &mut Accumulator);
    fn pairwise(acc: &Accumulator, out: &mut SideOut);
    fn dot<const N: usize>(x: &[u8; N], w: &[i8; N]) -> i32;
}

pub(super) struct Scalar;

#[inline(always)]
fn row_of(ft_w: &[i16], f: u16) -> &[i16; L1] {
    let i = f as usize * L1;
    ft_w[i..i + L1].try_into().unwrap()
}

impl Kernels for Scalar {
    #[inline(always)]
    fn refresh(ft_b: &Accumulator, ft_w: &[i16], feats: &[u16], acc: &mut Accumulator) {
        acc.0 = ft_b.0;
        for &f in feats {
            for (a, w) in acc.0.iter_mut().zip(row_of(ft_w, f)) {
                *a = a.wrapping_add(*w);
            }
        }
    }

    #[inline(always)]
    fn pairwise(acc: &Accumulator, out: &mut SideOut) {
        let (a, b) = acc.0.split_at(HALF);
        for i in 0..HALF {
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

/// Explicit AVX2 kernels. The accumulator is kept in registers while feature rows are added (two tiles of
/// 8 × 16 lanes), the pairwise product uses `mullo` + shift + `packus`, and the dot products use
/// `maddubs` (u8 × i8 → i16 pairs; inputs ≤ 127 so the pair sums cannot saturate) + `madd`.
#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
    use std::arch::x86_64::*;

    use super::super::net::{self, Network};
    use super::*;

    const TILE: usize = 128; // i16 lanes per register tile (8 registers)

    /// Only used by the `#[target_feature(enable = "avx2")]` entry points below, so its kernels always run
    /// with AVX2 available.
    struct Avx2;

    impl Kernels for Avx2 {
        #[inline(always)]
        fn refresh(ft_b: &Accumulator, ft_w: &[i16], feats: &[u16], acc: &mut Accumulator) {
            // SAFETY: see `Avx2`.
            unsafe { refresh(ft_b, ft_w, feats, acc) }
        }

        #[inline(always)]
        fn pairwise(acc: &Accumulator, out: &mut SideOut) {
            // SAFETY: see `Avx2`.
            unsafe { pairwise(acc, out) }
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
    fn refresh(ft_b: &Accumulator, ft_w: &[i16], feats: &[u16], acc: &mut Accumulator) {
        for f in feats {
            assert!((*f as usize + 1) * L1 <= ft_w.len());
        }
        for t in 0..L1 / TILE {
            // SAFETY: every row index was bounds-checked above; unaligned loads/stores.
            unsafe {
                let bp = ft_b.0.as_ptr().add(t * TILE) as *const __m256i;
                let mut r: [__m256i; 8] = std::array::from_fn(|k| _mm256_loadu_si256(bp.add(k)));
                for &f in feats {
                    let wp = ft_w.as_ptr().add(f as usize * L1 + t * TILE) as *const __m256i;
                    for (k, reg) in r.iter_mut().enumerate() {
                        *reg = _mm256_add_epi16(*reg, _mm256_loadu_si256(wp.add(k)));
                    }
                }
                let ap = acc.0.as_mut_ptr().add(t * TILE) as *mut __m256i;
                for (k, reg) in r.iter().enumerate() {
                    _mm256_storeu_si256(ap.add(k), *reg);
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    fn pairwise(acc: &Accumulator, out: &mut SideOut) {
        // SAFETY: fixed-size arrays; HALF is a multiple of 32.
        unsafe {
            let zero = _mm256_setzero_si256();
            let max = _mm256_set1_epi16(FT_Q as i16);
            let p = acc.0.as_ptr() as *const __m256i;
            let q = out.as_mut_ptr() as *mut __m256i;
            let prod = |i: usize| {
                let a = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(i)), zero), max);
                let b = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(HALF / 16 + i)), zero), max);
                _mm256_srli_epi16::<9>(_mm256_mullo_epi16(a, b))
            };
            for j in 0..HALF / 32 {
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
