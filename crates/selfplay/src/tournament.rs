//! Round robin over a roster: every matchup plays the same `pairs` seeds (common random numbers), both
//! seatings each. Output: Bradley-Terry Elo with bootstrap CIs, per-bot totals, and the score matrix.

use serde::Serialize;
use tetris_bot::BotSpec;

use crate::conditions::Conditions;
use crate::rating::{Outcome, Ratings, ratings};
use crate::report::{BotTotals, totals};
use crate::schedule::{Job, PairResult, derive_seed, run};
use crate::stats::pair_score;

/// Roster names: each bot's name, with `#2`, `#3`, … appended to repeats.
pub fn unique_names(bots: &[BotSpec]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for b in bots {
        let (mut n, mut k) = (b.name.clone(), 2);
        while names.contains(&n) {
            n = format!("{}#{k}", b.name);
            k += 1;
        }
        names.push(n);
    }
    names
}

#[derive(Clone, Debug)]
pub struct Tournament {
    /// [`unique_names`] of `bots`, or any other distinct names.
    pub names: Vec<String>,
    pub bots: Vec<BotSpec>,
    pub pairs: usize,
    pub seed: u64,
    pub cond: Conditions,
    /// Roster index whose Elo is 0.
    pub anchor: usize,
    pub bootstrap: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Matchup {
    pub a: String,
    pub b: String,
    /// A's mean score per game (draws ½) and a 95% normal interval over seed pairs ([`pair_score`]).
    pub score: f64,
    pub lo: f64,
    pub hi: f64,
    pub games: usize,
    pub draws: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TournamentResult {
    pub ratings: Ratings,
    pub totals: Vec<(String, BotTotals)>,
    pub matchups: Vec<Matchup>,
    pub games: usize,
    pub wall_s: f64,
    #[serde(skip)]
    pub pairs: Vec<PairResult>,
}

pub fn run_tournament(t: &Tournament) -> TournamentResult {
    let n = t.bots.len();
    let mut jobs = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            for k in 0..t.pairs {
                let draw = t.cond.draw(derive_seed(&[t.seed, k as u64]));
                jobs.push(Job::new((a, &t.bots[a]), (b, &t.bots[b]), &draw, t.cond.max_pieces, false));
            }
        }
    }
    let t0 = std::time::Instant::now();
    let pairs = run(&jobs);
    let wall_s = t0.elapsed().as_secs_f64();
    let outcomes: Vec<Outcome> =
        pairs.iter().map(|p| Outcome { a: p.a, b: p.b, score: p.a_total(), games: 2.0 }).collect();
    let r = ratings(&t.names, &outcomes, t.anchor, t.bootstrap, t.seed ^ 0x5eed);
    let tot = totals(n, &pairs);
    let mut matchups = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let ps: Vec<&PairResult> = pairs.iter().filter(|p| p.a == a && p.b == b).collect();
            let (score, lo, hi) = pair_score(&ps);
            let draws = ps.iter().flat_map(|p| &p.games).filter(|g| g.winner.is_none()).count();
            matchups.push(Matchup {
                a: t.names[a].clone(),
                b: t.names[b].clone(),
                score,
                lo,
                hi,
                games: 2 * ps.len(),
                draws,
            });
        }
    }
    TournamentResult {
        ratings: r,
        totals: t.names.iter().cloned().zip(tot).collect(),
        matchups,
        games: 2 * jobs.len(),
        wall_s,
        pairs,
    }
}

/// Text report: Elo table (sorted), matchups, per-bot totals.
pub fn render(t: &TournamentResult) -> String {
    let mut s = format!(
        "Bradley-Terry Elo (anchor {} = 0; 95% bootstrap CI, {} resamples of seed pairs)\n",
        t.ratings.anchor, t.ratings.bootstrap
    );
    let mut rs: Vec<_> = t.ratings.ratings.iter().collect();
    rs.sort_by(|a, b| b.elo.total_cmp(&a.elo));
    for r in rs {
        s += &format!(
            "  {:<24} {:>7.1}  [{:>7.1}, {:>7.1}]  score {:.3} over {} games\n",
            r.name, r.elo, r.lo, r.hi, r.score, r.games
        );
    }
    s += "\nMatchups (A's score per game, draws ½; 95% CI over seed pairs)\n";
    for m in &t.matchups {
        s += &format!(
            "  {:<24} vs {:<24} {:.3} [{:.3}, {:.3}]  {} games, {} draws\n",
            m.a, m.b, m.score, m.lo, m.hi, m.games, m.draws
        );
    }
    let (names, tot): (Vec<String>, Vec<BotTotals>) = t.totals.iter().cloned().unzip();
    s += "\n";
    s += &crate::report::table(&names, &tot);
    s += &format!("{} games in {:.1} s wall\n", t.games, t.wall_s);
    s
}
