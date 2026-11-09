//! Paired-seed jobs: each [`Job`] is one pair draw played twice with the seats swapped. Games are run in
//! parallel (rayon, one task per game for load balance) and regrouped into pairs.

use rayon::prelude::*;
use tetris_bot::BotSpec;
use tetris_sim::SplitMix;

use crate::conditions::PairDraw;
use crate::game::{GameResult, GameSpec, play_game};

/// Mix `parts` into one seed (SplitMix over a running hash), so seeds are independent per component.
pub fn derive_seed(parts: &[u64]) -> u64 {
    let mut h = 0x243f_6a88_85a3_08d3u64;
    for &p in parts {
        h = SplitMix(h ^ p.wrapping_mul(0x9e37_79b9_7f4a_7c15)).next_u64();
    }
    h
}

/// One pair: bot `a` vs bot `b` (indices into the caller's roster), `games[0]` = A in seat 0.
#[derive(Clone, Debug)]
pub struct Job {
    pub a: usize,
    pub b: usize,
    pub games: [GameSpec; 2],
}

impl Job {
    /// Both games of a pair: the same match seed and rules; each bot keeps its own frames per piece and
    /// RNG seed across the swap, so the seating is the only difference.
    pub fn new(a: (usize, &BotSpec), b: (usize, &BotSpec), draw: &PairDraw, max_pieces: u32, record: bool) -> Job {
        let game = |swapped| draw.game(a.1, b.1, swapped, max_pieces, record);
        Job { a: a.0, b: b.0, games: [game(false), game(true)] }
    }
}

#[derive(Clone, Debug)]
pub struct PairResult {
    pub a: usize,
    pub b: usize,
    /// `[A in seat 0, A in seat 1]`.
    pub games: [GameResult; 2],
}

impl PairResult {
    /// A's score in game `g` (1 win, ½ draw, 0 loss).
    pub fn a_score(&self, g: usize) -> f64 {
        match self.games[g].winner {
            Some(w) if w == g => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        }
    }

    /// A's total over both games (0, ½, …, 2).
    pub fn a_total(&self) -> f64 {
        self.a_score(0) + self.a_score(1)
    }
}

/// Run every job; results in job order.
pub fn run(jobs: &[Job]) -> Vec<PairResult> {
    let mut games: Vec<Option<GameResult>> =
        (0..2 * jobs.len()).into_par_iter().map(|k| Some(play_game(&jobs[k / 2].games[k % 2]))).collect();
    jobs.iter()
        .enumerate()
        .map(|(i, j)| PairResult {
            a: j.a,
            b: j.b,
            games: [games[2 * i].take().unwrap(), games[2 * i + 1].take().unwrap()],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Conditions;
    use tetris_bot::{SearchOptions, weights};

    #[test]
    fn pairs_swap_seats_only() {
        let opts = SearchOptions { width: 4, ..Default::default() };
        let a = BotSpec::hand(weights::preset("default").unwrap(), opts);
        let b = BotSpec::hand(weights::preset("steady").unwrap(), opts);
        let draw = Conditions::default().draw(derive_seed(&[1, 2]));
        let j = Job::new((0, &a), (1, &b), &draw, 100, false);
        let [g0, g1] = &j.games;
        assert_eq!(g0.seed, g1.seed);
        assert_eq!(g0.fpp, [g1.fpp[1], g1.fpp[0]]);
        assert_eq!(g0.bot_seeds, [g1.bot_seeds[1], g1.bot_seeds[0]]);
        assert_eq!(g0.bots[0], g1.bots[1]);
        assert!(std::ptr::eq(g0.cfg, g1.cfg));
        assert_ne!(derive_seed(&[1, 2]), derive_seed(&[2, 1]));
    }
}
