//! Random number generators.
//!
//! - [`Rng`]: TETR.IO's Park–Miller "minimal standard" LCG (`src/engine/utils/rng/index.ts`, RULES.md §11),
//!   which drives the bag and the garbage holes.
//! - [`SplitMix`]: SplitMix64, for everything that is not the game itself (bots, match scheduling,
//!   tuning, test and benchmark fixtures).

const MODULUS: u64 = 2_147_483_647;
const MULTIPLIER: u64 = 16_807;
const MAX_FLOAT: f64 = 2_147_483_646.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rng {
    value: u32,
}

impl Rng {
    /// `new RNG(seed)`: `v = seed % 2147483647; if (v <= 0) v += 2147483646`.
    pub fn new(seed: i64) -> Rng {
        let mut v = seed % MODULUS as i64; // JS `%` keeps the dividend's sign, like Rust's.
        if v <= 0 {
            v += 2_147_483_646;
        }
        Rng { value: v as u32 }
    }

    /// Internal value (`RNG.seed` getter).
    #[inline]
    pub fn state(&self) -> u32 {
        self.value
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.value = (MULTIPLIER * self.value as u64 % MODULUS) as u32;
        self.value
    }

    /// `(next() - 1) / 2147483646`, in [0, 1).
    #[inline]
    pub fn next_float(&mut self) -> f64 {
        (self.next_u32() - 1) as f64 / MAX_FLOAT
    }

    /// `floor(nextFloat() * n)`, computed in f64 exactly like the engine.
    #[inline]
    pub fn next_below(&mut self, n: u32) -> u32 {
        (self.next_float() * n as f64).floor() as u32
    }

    /// `shuffleArray`: Fisher–Yates from the end.
    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        if a.is_empty() {
            return;
        }
        let mut i = a.len() - 1;
        while i != 0 {
            let r = self.next_below(i as u32 + 1) as usize;
            a.swap(i, r);
            i -= 1;
        }
    }
}

/// A valid Park–Miller seed (`1..=2^31 - 2`) from any `u64`.
#[inline]
pub fn park_miller_seed(x: u64) -> i64 {
    (x % 2_147_483_646) as i64 + 1
}

/// SplitMix64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix(pub u64);

impl SplitMix {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0; the modulo bias is negligible here).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    #[inline]
    pub fn park_miller_seed(&mut self) -> i64 {
        park_miller_seed(self.next_u64())
    }

    /// Uniform in [0, 1) with 24 bits.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [0, 1) with 53 bits.
    #[inline]
    pub fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_normalisation() {
        assert_eq!(Rng::new(0).state(), 2_147_483_646);
        assert_eq!(Rng::new(2_147_483_647).state(), 2_147_483_646);
        assert_eq!(Rng::new(5).state(), 5);
        assert_eq!(Rng::new(-5).state(), 2_147_483_641);
    }

    #[test]
    fn known_sequence() {
        let mut r = Rng::new(1);
        assert_eq!(r.next_u32(), 16807);
        assert_eq!(r.next_u32(), 282_475_249);
        assert_eq!(r.next_u32(), 1_622_650_073);
    }
}
