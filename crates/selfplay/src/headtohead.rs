//! A paired-seed match between two bots at fixed speeds (`sim bot-vs-bot`): pair `i` plays one draw of
//! [`FixedSpeed`] twice with the seats swapped.

use tetris_bot::BotSpec;

use crate::conditions::FixedSpeed;
use crate::report::{BotTotals, totals};
use crate::schedule::{Job, PairResult, run};
use crate::stats::{binomial_two_sided, wilson};

#[derive(Clone, Debug)]
pub struct HeadToHead {
    pub a: BotSpec,
    pub b: BotSpec,
    pub pairs: usize,
    pub seed: u64,
    pub speed: FixedSpeed,
    /// Record the first `record` pairs as oracle requests.
    pub record: usize,
}

impl HeadToHead {
    pub fn jobs(&self) -> Vec<Job> {
        (0..self.pairs)
            .map(|i| {
                let draw = self.speed.draw(self.seed ^ (i as u64).wrapping_mul(0xa076_1d64_78bd_642f));
                Job::new((0, &self.a), (1, &self.b), &draw, i < self.record)
            })
            .collect()
    }

    /// Every pair (A is roster index 0, B is 1).
    pub fn run(&self) -> Vec<PairResult> {
        run(&self.jobs())
    }
}

/// Totals per bot, plus the pair-level counts for the sign test.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub a: BotTotals,
    pub b: BotTotals,
    /// Pairs A scored more in / B scored more in / tied (draws count ½).
    pub pairs_a: u64,
    pub pairs_b: u64,
    pub pairs_tied: u64,
}

impl Summary {
    pub fn from_pairs(pairs: &[PairResult]) -> Summary {
        let [a, b]: [BotTotals; 2] = totals(2, pairs).try_into().unwrap();
        let mut s = Summary { a, b, pairs_a: 0, pairs_b: 0, pairs_tied: 0 };
        for p in pairs {
            match p.a_total().total_cmp(&1.0) {
                std::cmp::Ordering::Greater => s.pairs_a += 1,
                std::cmp::Ordering::Less => s.pairs_b += 1,
                std::cmp::Ordering::Equal => s.pairs_tied += 1,
            }
        }
        s
    }

    /// A's win rate over decisive games and its Wilson 95% CI.
    pub fn win_rate(&self) -> (f64, f64, f64) {
        let n = self.a.wins + self.b.wins;
        let (lo, hi) = wilson(self.a.wins, n, 1.96);
        (self.a.wins as f64 / n.max(1) as f64, lo, hi)
    }

    /// Sign test on decisive pairs (two-sided).
    pub fn pair_p(&self) -> f64 {
        binomial_two_sided(self.pairs_a, self.pairs_a + self.pairs_b)
    }

    /// Binomial test on decisive games (two-sided; ignores the pairing).
    pub fn game_p(&self) -> f64 {
        binomial_two_sided(self.a.wins, self.a.wins + self.b.wins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_bot::{SearchOptions, weights};

    #[test]
    fn summary_counts_every_game() {
        let opts = SearchOptions { width: 8, ..Default::default() };
        let h = HeadToHead {
            a: BotSpec::hand(weights::preset("default").unwrap(), opts),
            b: BotSpec::hand(weights::preset("steady").unwrap(), opts),
            pairs: 2,
            seed: 9,
            speed: FixedSpeed { fpp: [24, 30], jitter: 1, max_pieces: 60 },
            record: 1,
        };
        let res = h.run();
        assert!(res[0].games.iter().all(|g| g.request.is_some()) && res[1].games.iter().all(|g| g.request.is_none()));
        let s = Summary::from_pairs(&res);
        assert_eq!(s.a.games, 4);
        assert_eq!(s.a.wins + s.b.wins + s.a.draws, 4);
        assert_eq!(s.pairs_a + s.pairs_b + s.pairs_tied, 2);
    }
}
