//! Bradley-Terry ratings on the Elo scale, with bootstrap confidence intervals.
//!
//! Model: `P(i beats j) = p_i / (p_i + p_j)`, draws count as half a win. Fitted with Hunter's MM
//! iteration; a weak prior (one virtual drawn game per matchup that was played) keeps the fit finite when a
//! bot wins or loses every game. `elo_i = 400 · log10(p_i)`, shifted so the anchor is 0.
//!
//! CIs: resample the paired units (both seatings of one seed) with replacement **within each matchup**,
//! refit, and take the 2.5% / 97.5% percentiles of each bot's Elo relative to the anchor.

use std::collections::BTreeMap;

use serde::Serialize;
use tetris_sim::SplitMix;

/// One unit of evidence: `a` scored `score` points out of `games` against `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    pub a: usize,
    pub b: usize,
    pub score: f64,
    pub games: f64,
}

/// Elo per player (anchor = 0).
pub fn bradley_terry(n: usize, outcomes: &[Outcome], anchor: usize) -> Vec<f64> {
    let mut wins = vec![0.0; n];
    let mut games = vec![vec![0.0; n]; n];
    for o in outcomes {
        wins[o.a] += o.score;
        wins[o.b] += o.games - o.score;
        games[o.a][o.b] += o.games;
        games[o.b][o.a] += o.games;
    }
    for i in 0..n {
        for j in i + 1..n {
            if games[i][j] > 0.0 {
                games[i][j] += 1.0;
                games[j][i] += 1.0;
                wins[i] += 0.5;
                wins[j] += 0.5;
            }
        }
    }
    let mut p = vec![1.0f64; n];
    for _ in 0..10_000 {
        let mut next = vec![0.0; n];
        for i in 0..n {
            let denom: f64 = (0..n).filter(|&j| games[i][j] > 0.0).map(|j| games[i][j] / (p[i] + p[j])).sum();
            next[i] = if denom > 0.0 { wins[i] / denom } else { p[i] };
        }
        let mean_log = next.iter().map(|x: &f64| x.ln()).sum::<f64>() / n as f64;
        let mut delta = 0.0f64;
        for i in 0..n {
            let v = (next[i].ln() - mean_log).exp();
            delta = delta.max((v.ln() - p[i].ln()).abs());
            p[i] = v;
        }
        if delta < 1e-10 {
            break;
        }
    }
    let base = p[anchor].log10();
    p.iter().map(|x| 400.0 * (x.log10() - base)).collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct Rating {
    pub name: String,
    pub elo: f64,
    pub lo: f64,
    pub hi: f64,
    pub games: f64,
    /// Mean score per game (draws ½) against everyone.
    pub score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Ratings {
    pub anchor: String,
    pub bootstrap: usize,
    pub ratings: Vec<Rating>,
}

/// Fit, then bootstrap `samples` times (seeded; deterministic).
pub fn ratings(names: &[String], outcomes: &[Outcome], anchor: usize, samples: usize, seed: u64) -> Ratings {
    let n = names.len();
    let elo = bradley_terry(n, outcomes, anchor);
    let mut groups: BTreeMap<(usize, usize), Vec<Outcome>> = BTreeMap::new();
    for o in outcomes {
        groups.entry((o.a.min(o.b), o.a.max(o.b))).or_default().push(*o);
    }
    let mut rng = SplitMix(seed);
    let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); n];
    let mut sample = Vec::with_capacity(outcomes.len());
    for _ in 0..samples {
        sample.clear();
        for g in groups.values() {
            for _ in 0..g.len() {
                sample.push(g[rng.below(g.len() as u64) as usize]);
            }
        }
        for (i, e) in bradley_terry(n, &sample, anchor).into_iter().enumerate() {
            draws[i].push(e);
        }
    }
    let mut totals = vec![(0.0, 0.0); n];
    for o in outcomes {
        totals[o.a].0 += o.score;
        totals[o.a].1 += o.games;
        totals[o.b].0 += o.games - o.score;
        totals[o.b].1 += o.games;
    }
    let ratings = (0..n)
        .map(|i| {
            let d = &mut draws[i];
            d.sort_by(f64::total_cmp);
            let q = |f: f64| if d.is_empty() { elo[i] } else { d[((d.len() - 1) as f64 * f).round() as usize] };
            Rating {
                name: names[i].clone(),
                elo: elo[i],
                lo: q(0.025),
                hi: q(0.975),
                games: totals[i].1,
                score: totals[i].0 / totals[i].1.max(1.0),
            }
        })
        .collect();
    Ratings { anchor: names[anchor].clone(), bootstrap: samples, ratings }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic round robins with known strengths (Elo 0, 100, 200, 400), 20 seeds: the fit is unbiased
    /// and the 95% bootstrap CIs cover the truth about 95% of the time.
    #[test]
    fn recovers_known_strengths() {
        let truth = [0.0, 100.0, 200.0, 400.0];
        let (mut err, mut miss) = ([0.0; 4], 0);
        for seed in 0..20u64 {
            let mut rng = SplitMix(seed * 7919 + 3);
            let mut out = Vec::new();
            for a in 0..4 {
                for b in a + 1..4 {
                    let pa = 1.0 / (1.0 + 10f64.powf((truth[b] - truth[a]) / 400.0));
                    for _ in 0..4000 {
                        let s = rng.unit_f64() < pa;
                        out.push(Outcome { a, b, score: s as u8 as f64, games: 1.0 });
                    }
                }
            }
            let r = ratings(&["a", "b", "c", "d"].map(String::from), &out, 0, 200, seed);
            assert_eq!((r.ratings[0].elo, r.ratings[0].lo, r.ratings[0].hi), (0.0, 0.0, 0.0));
            for i in 1..4 {
                let x = &r.ratings[i];
                assert!(x.lo <= x.elo && x.elo <= x.hi && (x.elo - truth[i]).abs() < 30.0, "{i}: {x:?}");
                err[i] += (x.elo - truth[i]) / 20.0;
                miss += !(x.lo..=x.hi).contains(&truth[i]) as u32;
            }
        }
        assert!(err.iter().all(|e| e.abs() < 5.0), "mean error {err:?}");
        assert!(miss <= 8, "{miss}/60 CIs miss the truth");
    }

    #[test]
    fn perfect_record_stays_finite() {
        let out: Vec<_> = (0..50).map(|_| Outcome { a: 0, b: 1, score: 2.0, games: 2.0 }).collect();
        let e = bradley_terry(2, &out, 1);
        assert!(e[0].is_finite() && e[0] > 500.0);
    }
}
