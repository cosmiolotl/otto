//! What a pair of games is played under, drawn once per pair from a seed: the match seed, each bot's
//! speed and RNG seed, and the rules. Both seatings of the pair share the draw ([`crate::Job`]).
//!
//! - [`Conditions`]: varied speeds and rules (tournaments, tuning, training data).
//! - [`FixedSpeed`]: exact TL rules at a fixed speed per bot (`sim bot-vs-bot`).

use serde::{Deserialize, Serialize};
use tetris_bot::BotSpec;
use tetris_sim::{Config, SplitMix};

use crate::game::GameSpec;

/// Frames per piece at `pps` pieces per second (60 fps).
pub fn fpp_of_pps(pps: f64) -> u32 {
    (60.0 / pps).round().max(1.0) as u32
}

/// One pair's draw, per bot (A, B).
#[derive(Clone, Copy, Debug)]
pub struct PairDraw {
    pub match_seed: i64,
    pub fpp: [u32; 2],
    pub bot_seeds: [u64; 2],
    pub cfg: &'static Config,
}

impl PairDraw {
    /// The game with bot A in seat 0 (`swapped`: in seat 1). Each bot keeps its own speed and RNG seed.
    pub fn game(&self, a: &BotSpec, b: &BotSpec, swapped: bool, max_pieces: u32, record_keys: bool) -> GameSpec {
        let seat = |x: [u32; 2]| if swapped { [x[1], x[0]] } else { x };
        let bot_seeds = if swapped { [self.bot_seeds[1], self.bot_seeds[0]] } else { self.bot_seeds };
        GameSpec {
            seed: self.match_seed,
            bots: if swapped { [b.clone(), a.clone()] } else { [a.clone(), b.clone()] },
            fpp: seat(self.fpp),
            bot_seeds,
            max_pieces,
            record_keys,
            cfg: self.cfg,
        }
    }
}

/// Ranges that match conditions are drawn from, once per pair (the frames per piece follow the bot, so the
/// seat swap is the only difference between the two games).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Conditions {
    /// Pieces per second per bot, uniform in `[lo, hi]` (each bot independently).
    pub pps: [f64; 2],
    /// Fraction of pairs played under the exact TL rules; the rest vary garbage speed and messiness.
    pub tl_fraction: f64,
    /// Garbage speed (frames), multiples of 5 in `[lo, hi]`.
    pub garbage_speed: [u32; 2],
    /// `messiness_change` / `messiness_within`, steps of 0.1 in `[lo, hi]`.
    pub messiness_change: [f64; 2],
    pub messiness_within: [f64; 2],
    /// Total placements (both players) before a game is a draw.
    pub max_pieces: u32,
}

impl Default for Conditions {
    fn default() -> Self {
        Conditions {
            pps: [2.0, 3.0],
            tl_fraction: 0.5,
            garbage_speed: [10, 30],
            messiness_change: [0.5, 1.0],
            messiness_within: [0.0, 0.3],
            max_pieces: 1600,
        }
    }
}

impl Conditions {
    /// Exact TL rules at a fixed PPS (no variation).
    pub fn tl(pps: f64, max_pieces: u32) -> Conditions {
        Conditions { pps: [pps, pps], tl_fraction: 1.0, max_pieces, ..Default::default() }
    }

    pub fn draw(&self, seed: u64) -> PairDraw {
        let mut rng = SplitMix(seed);
        let match_seed = rng.park_miller_seed();
        let (fpp, cfg) = self.sample(&mut rng);
        let bot_seeds = [rng.next_u64(), rng.next_u64()];
        PairDraw { match_seed, fpp, bot_seeds, cfg }
    }

    /// Frames per piece of bot A and bot B, and the config.
    pub fn sample(&self, rng: &mut SplitMix) -> ([u32; 2], &'static Config) {
        let mut fpp = || fpp_of_pps(self.pps[0] + (self.pps[1] - self.pps[0]) * rng.unit_f64());
        let f = [fpp(), fpp()];
        if rng.unit_f64() < self.tl_fraction {
            return (f, Config::tl());
        }
        // Quantised so that the interned config pool stays small.
        let steps = |lo: f64, hi: f64, rng: &mut SplitMix| {
            let (a, b) = ((lo * 10.0).round() as u64, (hi * 10.0).round() as u64);
            (a + rng.below(b.saturating_sub(a) + 1)) as f64 / 10.0
        };
        let (gl, gh) = (self.garbage_speed[0].div_ceil(5), self.garbage_speed[1] / 5);
        let speed = 5 * (gl + rng.below(gh.saturating_sub(gl) as u64 + 1) as u32);
        let cfg = Config {
            garbage_speed: speed,
            messiness_change: steps(self.messiness_change[0], self.messiness_change[1], rng),
            messiness_within: steps(self.messiness_within[0], self.messiness_within[1], rng),
            ..Config::TL
        };
        (f, cfg.intern())
    }
}

/// TL rules, bot A at `fpp[0]` and bot B at `fpp[1]` frames per piece, each offset per pair by a uniform
/// `-jitter..=jitter` (de-mirroring: both players get the same pieces).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedSpeed {
    pub fpp: [u32; 2],
    pub jitter: u32,
}

impl FixedSpeed {
    pub fn draw(&self, seed: u64) -> PairDraw {
        let mut rng = SplitMix(seed);
        let match_seed = rng.park_miller_seed();
        let mut jit = |f: u32| {
            let j = self.jitter as i64;
            let d = if j == 0 { 0 } else { rng.below(2 * j as u64 + 1) as i64 - j };
            (f as i64 + d).max(1) as u32
        };
        let fpp = [jit(self.fpp[0]), jit(self.fpp[1])];
        let bot_seeds = [rng.next_u64(), rng.next_u64()];
        PairDraw { match_seed, fpp, bot_seeds, cfg: Config::tl() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varied_conditions_are_quantised_and_in_range() {
        let c = Conditions { tl_fraction: 0.0, ..Default::default() };
        let mut rng = SplitMix(7);
        for _ in 0..500 {
            let (f, cfg) = c.sample(&mut rng);
            assert!(f.iter().all(|&f| (20..=30).contains(&f)), "{f:?}");
            assert!(cfg.garbage_speed % 5 == 0 && (10..=30).contains(&cfg.garbage_speed));
            assert!((0.5..=1.0).contains(&cfg.messiness_change) && (0.0..=0.3).contains(&cfg.messiness_within));
        }
    }

    #[test]
    fn fixed_speed_jitters_within_bounds() {
        let s = FixedSpeed { fpp: [24, 30], jitter: 1 };
        for seed in 0..50 {
            let d = s.draw(seed);
            assert!((23..=25).contains(&d.fpp[0]) && (29..=31).contains(&d.fpp[1]), "{:?}", d.fpp);
        }
        assert_eq!(FixedSpeed { jitter: 0, ..s }.draw(3).fpp, [24, 30]);
    }
}
