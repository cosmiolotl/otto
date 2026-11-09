//! Statistics helpers for match reports.

use crate::schedule::PairResult;

/// Wilson score interval for `k` successes in `n` trials (`z = 1.96` for 95%). `(0, 1)` if `n == 0`.
pub fn wilson(k: u64, n: u64, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n_f = n as f64;
    let p = k as f64 / n_f;
    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let centre = (p + z2 / (2.0 * n_f)) / denom;
    let half = z * (p * (1.0 - p) / n_f + z2 / (4.0 * n_f * n_f)).sqrt() / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

fn ln_choose(n: u64, k: u64) -> f64 {
    let k = k.min(n - k);
    (0..k).map(|i| ((n - i) as f64).ln() - ((i + 1) as f64).ln()).sum()
}

/// Exact two-sided binomial test of `k` successes in `n` trials against p = 1/2 (the sign test):
/// `min(1, 2 · min(P(X ≤ k), P(X ≥ k)))`. `1.0` if `n == 0`.
pub fn binomial_two_sided(k: u64, n: u64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let ln_half_n = n as f64 * 0.5f64.ln();
    let pmf = |i: u64| (ln_choose(n, i) + ln_half_n).exp();
    let lower: f64 = (0..=k).map(pmf).sum();
    let upper: f64 = (k..=n).map(pmf).sum();
    (2.0 * lower.min(upper)).min(1.0)
}

/// A's mean per-game score over `pairs` and a 95% normal interval using the per-pair variance (pairs are
/// the independent unit; a pair's two games share the seed).
pub fn pair_score(pairs: &[&PairResult]) -> (f64, f64, f64) {
    let n = pairs.len() as f64;
    if pairs.is_empty() {
        return (0.5, 0.0, 1.0);
    }
    let xs: Vec<f64> = pairs.iter().map(|p| p.a_total() / 2.0).collect();
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let half = 1.96 * (var / n).sqrt();
    (mean, (mean - half).max(0.0), (mean + half).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let (lo, hi) = wilson(60, 100, 1.96);
        assert!((lo - 0.5020).abs() < 1e-3 && (hi - 0.6906).abs() < 1e-3, "{lo} {hi}");
        assert_eq!(binomial_two_sided(5, 10), 1.0);
        // P(X >= 9 | n = 10) = 11/1024, two-sided 22/1024.
        assert!((binomial_two_sided(9, 10) - 22.0 / 1024.0).abs() < 1e-12);
        assert!((binomial_two_sided(1, 10) - 22.0 / 1024.0).abs() < 1e-12);
        assert!(binomial_two_sided(600, 1000) < 1e-9);
    }
}
