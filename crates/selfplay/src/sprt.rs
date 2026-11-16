//! Sequential probability ratio test on paired-seed results, as fishtest gates Stockfish patches.
//!
//! Each seed pair (both seatings) gives A a score in {0, ½, 1, 1½, 2} (draws count ½); the five counts are
//! the *pentanomial*. The test is the normal-approximation GSPRT of fishtest's `LLR_logistic`: with `n`
//! games, mean per-game score `s` and per-game variance `σ²` estimated from the pair scores,
//! `LLR = n (s₁ − s₀)(2s − s₀ − s₁) / (2σ²)`, where `sᵢ` is the expected score at logistic Elo `eloᵢ`.
//! H0 (elo = elo0) is accepted when the LLR falls below `ln(β/(1−α))`, H1 (elo = elo1) when it rises above
//! `ln((1−β)/α)`.

use crate::schedule::PairResult;

/// Pairs by A's pair score: `counts[k]` pairs scored `k/2` (0 = lost both, 4 = won both).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pentanomial {
    pub counts: [u64; 5],
}

impl Pentanomial {
    pub fn from_pairs<'a>(pairs: impl IntoIterator<Item = &'a PairResult>) -> Pentanomial {
        let mut p = Pentanomial::default();
        for r in pairs {
            p.add(r.a_total());
        }
        p
    }

    /// Add one pair with A's total `score` over its two games (0, ½, …, 2).
    pub fn add(&mut self, score: f64) {
        let k = (2.0 * score).round();
        assert!((0.0..=4.0).contains(&k), "pair score {score} not in [0, 2]");
        self.counts[k as usize] += 1;
    }

    pub fn pairs(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// `(games, mean per-game score, per-game variance)` from (possibly fractional) counts, as fishtest's
    /// `stats`: the variance of the pair score divided by 2 (a pair is two games).
    fn moments(counts: [f64; 5]) -> (f64, f64, f64) {
        let pairs: f64 = counts.iter().sum();
        let games = 2.0 * pairs;
        let score = |k: usize| k as f64 / 2.0;
        let mu = (0..5).map(|k| counts[k] * score(k)).sum::<f64>() / games;
        let var = (0..5).map(|k| counts[k] * (score(k) - 2.0 * mu).powi(2)).sum::<f64>() / games;
        (games, mu, var)
    }

    /// A's mean per-game score (½ if empty).
    pub fn score(&self) -> f64 {
        if self.pairs() == 0 { 0.5 } else { Self::moments(self.counts.map(|c| c as f64)).1 }
    }

    /// Logistic Elo of A over B and its 95% normal interval from the pentanomial variance (the score's
    /// standard error is √(σ²/games)). Infinite when the score (or a bound) reaches 0 or 1.
    pub fn elo(&self) -> (f64, f64, f64) {
        if self.pairs() == 0 {
            return (0.0, f64::NEG_INFINITY, f64::INFINITY);
        }
        let (games, mu, var) = Self::moments(self.counts.map(|c| c as f64));
        let half = 1.96 * (var / games).sqrt();
        (elo_of_score(mu), elo_of_score(mu - half), elo_of_score(mu + half))
    }
}

/// Expected score at logistic Elo difference `elo`.
pub fn score_of_elo(elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo / 400.0))
}

/// Logistic Elo difference for expected score `s` (±∞ at or beyond 0 and 1).
pub fn elo_of_score(s: f64) -> f64 {
    if s <= 0.0 {
        f64::NEG_INFINITY
    } else if s >= 1.0 {
        f64::INFINITY
    } else {
        -400.0 * (1.0 / s - 1.0).log10()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// elo = elo0 accepted: the candidate is not better by elo1.
    H0,
    /// elo = elo1 accepted.
    H1,
}

/// The test: H0 elo = `elo0` against H1 elo = `elo1` (logistic Elo, `elo0 < elo1`), false-positive rate
/// `alpha`, false-negative rate `beta`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sprt {
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Sprt {
    pub fn new(elo0: f64, elo1: f64, alpha: f64, beta: f64) -> Result<Sprt, String> {
        if !elo0.is_finite() || !elo1.is_finite() || elo0 >= elo1 {
            return Err(format!("sprt: need elo0 < elo1, got {elo0}, {elo1}"));
        }
        for (name, x) in [("alpha", alpha), ("beta", beta)] {
            if !(x > 0.0 && x < 0.5) {
                return Err(format!("sprt: {name} {x} not in (0, 0.5)"));
            }
        }
        Ok(Sprt { elo0, elo1, alpha, beta })
    }

    /// `(lower, upper)`: accept H0 at or below `lower`, H1 at or above `upper`.
    pub fn bounds(&self) -> (f64, f64) {
        ((self.beta / (1.0 - self.alpha)).ln(), ((1.0 - self.beta) / self.alpha).ln())
    }

    /// The log-likelihood ratio of H1 over H0 (0 with no pairs). Empty pentanomial cells get a 1e-3 prior
    /// first, as fishtest's `regularize`, so a one-sided result (every pair won 2–0) has finite variance.
    pub fn llr(&self, p: &Pentanomial) -> f64 {
        if p.pairs() == 0 {
            return 0.0;
        }
        let counts = p.counts.map(|c| if c == 0 { 1e-3 } else { c as f64 });
        let (games, mu, var) = Pentanomial::moments(counts);
        let (s0, s1) = (score_of_elo(self.elo0), score_of_elo(self.elo1));
        games * (s1 - s0) * (2.0 * mu - s0 - s1) / var / 2.0
    }

    /// The decision at `llr`, if a bound is crossed.
    pub fn decide(&self, llr: f64) -> Option<Decision> {
        let (lo, hi) = self.bounds();
        if llr >= hi {
            Some(Decision::H1)
        } else if llr <= lo {
            Some(Decision::H0)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn penta(counts: [u64; 5]) -> Pentanomial {
        Pentanomial { counts }
    }

    #[test]
    fn llr_known_value() {
        // 100 pairs: games 200, mean 0.6, per-game variance 26/200 = 0.13; s1 = L(10) = 0.514387...
        // LLR = 200 · (s1 − ½)(1.2 − ½ − s1) / 0.13 / 2.
        let s = Sprt::new(0.0, 10.0, 0.05, 0.05).unwrap();
        let llr = s.llr(&penta([5, 10, 40, 30, 15]));
        assert!((llr - 2.054_189_05).abs() < 1e-6, "{llr}");
        assert_eq!(s.decide(llr), None);
        assert!((score_of_elo(10.0) - 0.514_387_184_2).abs() < 1e-9);
    }

    #[test]
    fn symmetric_results_are_neutral() {
        let s = Sprt::new(-5.0, 5.0, 0.05, 0.05).unwrap();
        for c in [[3, 10, 40, 10, 3], [0, 7, 12, 7, 0], [50, 0, 0, 0, 50]] {
            assert!(s.llr(&penta(c)).abs() < 1e-9, "{c:?}");
            assert!(penta(c).elo().0.abs() < 1e-9);
        }
        assert_eq!(s.llr(&Pentanomial::default()), 0.0);
    }

    #[test]
    fn bounds_and_decisions() {
        let s = Sprt::new(0.0, 10.0, 0.05, 0.05).unwrap();
        let (lo, hi) = s.bounds();
        assert!((lo + 2.944_438_979).abs() < 1e-8 && (hi - 2.944_438_979).abs() < 1e-8);
        assert_eq!(s.decide(hi), Some(Decision::H1));
        assert_eq!(s.decide(lo - 0.1), Some(Decision::H0));
        assert_eq!(s.decide(0.0), None);
        // A clear win accepts H1, a clear loss H0, including the one-sided cases (regularized variance).
        assert_eq!(s.decide(s.llr(&penta([5, 10, 40, 30, 15]).scaled(2))), Some(Decision::H1));
        assert_eq!(s.decide(s.llr(&penta([0, 0, 0, 0, 20]))), Some(Decision::H1));
        assert_eq!(s.decide(s.llr(&penta([15, 30, 40, 10, 5]).scaled(2))), Some(Decision::H0));
        assert_eq!(s.decide(s.llr(&penta([20, 0, 0, 0, 0]))), Some(Decision::H0));
        // An even match is evidence against elo1 > 0 when elo0 = 0.
        assert!(s.llr(&penta([10, 20, 40, 20, 10])) < 0.0);
        assert!(Sprt::new(10.0, 0.0, 0.05, 0.05).is_err() && Sprt::new(0.0, 5.0, 0.0, 0.05).is_err());
    }

    #[test]
    fn elo_estimate() {
        // mean 0.6, se √(0.13/200): Elo 70.44 [34.88, 107.52].
        let (e, lo, hi) = penta([5, 10, 40, 30, 15]).elo();
        assert!(
            (e - 70.436_503_6).abs() < 1e-6 && (lo - 34.880_852_5).abs() < 1e-6 && (hi - 107.515_516_3).abs() < 1e-6
        );
        assert!((elo_of_score(score_of_elo(123.0)) - 123.0).abs() < 1e-9);
        let (e, _, hi) = penta([0, 0, 0, 0, 7]).elo();
        assert!(e == f64::INFINITY && hi == f64::INFINITY);
        let mut p = Pentanomial::default();
        for s in [0.0, 0.5, 1.0, 1.5, 2.0, 2.0] {
            p.add(s);
        }
        assert_eq!(p.counts, [1, 1, 1, 1, 2]);
    }

    impl Pentanomial {
        fn scaled(self, k: u64) -> Pentanomial {
            Pentanomial { counts: self.counts.map(|c| c * k) }
        }
    }
}
