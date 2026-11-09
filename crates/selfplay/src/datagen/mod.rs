//! Training data for the value network (`sim datagen`, `docs/NNUE.md`).
//!
//! Plays self-play games between bots drawn from a roster and records sampled decision points: both
//! players' `rows-v1` features (from the **real** states; the features read only visible information),
//! the layer-stack buckets, the teacher's search score and static evaluation, and, once the game is over,
//! its result from the side to move's point of view. Games are deterministic given the seed; batches of
//! games run in parallel and are written in game order, so a data set is reproducible.

pub mod format;

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use tetris_bot::nnue::{Features, bucket, extract};
use tetris_bot::{BotSpec, Evaluator, HandEval, PathAcc};
use tetris_sim::SplitMix;

use crate::conditions::{Conditions, PairDraw};
use crate::game::play_game_with;
use crate::schedule::derive_seed;
use format::{DataWriter, Sample};

#[derive(Clone, Debug)]
pub struct DatagenConfig {
    /// Each seat draws a bot uniformly from the roster (repeat an entry to weight it).
    pub bots: Vec<BotSpec>,
    pub games: usize,
    pub seed: u64,
    pub cond: Conditions,
    /// Record each decision with probability `1 / every`.
    pub every: u32,
    /// Output stem (`STEM.meta`, `STEM.feat`, `STEM.json`).
    pub out: PathBuf,
    /// Games per parallel batch.
    pub batch: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DatagenTotals {
    pub games: u64,
    pub samples: u64,
    pub wins: [u64; 2],
    pub draws: u64,
    pub pieces: u64,
}

/// The static evaluation recorded as the teacher signal: the seat's own hand-written weights, or the
/// default weights for a network bot.
fn teacher(spec: &BotSpec) -> HandEval {
    HandEval::new(spec.weights().cloned().unwrap_or_default())
}

/// One game's samples (results filled in), its winner and its total placements.
pub fn play_and_sample(cfg: &DatagenConfig, game: usize) -> (Vec<Sample>, Option<usize>, u32) {
    let mut rng = SplitMix(derive_seed(&[cfg.seed, game as u64, 0xda7a]));
    let n = cfg.bots.len() as u64;
    let seats = [rng.below(n) as usize, rng.below(n) as usize];
    let match_seed = rng.park_miller_seed();
    let (fpp, mcfg) = cfg.cond.sample(&mut rng);
    let draw = PairDraw { match_seed, fpp, bot_seeds: [rng.next_u64(), rng.next_u64()], cfg: mcfg };
    let spec = draw.game(&cfg.bots[seats[0]], &cfg.bots[seats[1]], false, cfg.cond.max_pieces, false);
    let teachers = [teacher(&spec.bots[0]), teacher(&spec.bots[1])];
    let mut out: Vec<(usize, Sample)> = Vec::new();
    let g = play_game_with(&spec, |m, p, d| {
        if rng.below(cfg.every.max(1) as u64) != 0 {
            return;
        }
        let (real, opp) = (&m.players[p], &m.players[1 - p]);
        let mut s = Sample {
            me: Features::default(),
            opp: Features::default(),
            bucket_me: bucket(real) as u8,
            bucket_opp: bucket(opp) as u8,
            result: 0,
            mode: d.mode.index() as u8,
            ply: real.stats.pieces.min(u16::MAX as u32) as u16,
            to_end: 0,
            search_score: d.score,
            static_score: teachers[p].eval(real, &PathAcc::default()),
            game: game as u32,
        };
        extract(real, fpp[p], &mut s.me);
        extract(opp, fpp[1 - p], &mut s.opp);
        out.push((p, s));
    });
    let finals = [g.stats[0].pieces, g.stats[1].pieces];
    let samples = out
        .into_iter()
        .map(|(p, mut s)| {
            s.result = match g.winner {
                Some(w) if w == p => 1,
                Some(_) => -1,
                None => 0,
            };
            s.to_end = (finals[p] - s.ply as u32).min(u16::MAX as u32) as u16;
            s
        })
        .collect();
    (samples, g.winner, finals[0] + finals[1])
}

/// Generate the whole data set; prints one progress line per batch.
pub fn run(cfg: &DatagenConfig) -> std::io::Result<DatagenTotals> {
    let settings = serde_json::json!({
        "games": cfg.games,
        "seed": cfg.seed,
        "every": cfg.every,
        "conditions": serde_json::to_value(&cfg.cond).unwrap(),
        "bots": cfg.bots.iter().map(|b| serde_json::json!({ "name": b.name, "width": b.opts.width })).collect::<Vec<_>>(),
    });
    let mut w = DataWriter::create(&cfg.out, settings)?;
    let mut tot = DatagenTotals::default();
    let t0 = Instant::now();
    let batch = cfg.batch.max(1);
    for start in (0..cfg.games).step_by(batch) {
        let end = (start + batch).min(cfg.games);
        let res: Vec<_> = (start..end).into_par_iter().map(|g| play_and_sample(cfg, g)).collect();
        for (samples, winner, pieces) in res {
            for s in &samples {
                w.push(s)?;
            }
            tot.games += 1;
            tot.samples += samples.len() as u64;
            tot.pieces += pieces as u64;
            match winner {
                Some(p) => tot.wins[p] += 1,
                None => tot.draws += 1,
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "datagen: {}/{} games, {} samples, {:.0} pieces/game, {:.1} games/s, {:.0} samples/s",
            tot.games,
            cfg.games,
            tot.samples,
            tot.pieces as f64 / tot.games as f64,
            tot.games as f64 / secs,
            tot.samples as f64 / secs
        );
    }
    w.finish()?;
    Ok(tot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_bot::{SearchOptions, weights};

    #[test]
    fn deterministic_samples_with_results() {
        let opts = SearchOptions { width: 4, depth: 2, ..Default::default() };
        let cfg = DatagenConfig {
            bots: vec![
                BotSpec::hand(weights::preset("default").unwrap(), opts),
                BotSpec::hand(weights::preset("greedy-app").unwrap(), opts),
            ],
            games: 2,
            seed: 5,
            cond: Conditions { max_pieces: 300, ..Default::default() },
            every: 3,
            out: PathBuf::new(),
            batch: 2,
        };
        let (a, wa, pa) = play_and_sample(&cfg, 1);
        let (b, wb, pb) = play_and_sample(&cfg, 1);
        assert_eq!((wa, pa, a.len()), (wb, pb, b.len()));
        assert!(!a.is_empty());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.me.as_slice(), y.me.as_slice());
            assert_eq!((x.result, x.ply, x.to_end), (y.result, y.ply, y.to_end));
        }
        // A draw labels every sample 0; a decided game labels every sample a win or a loss.
        assert!(a.iter().all(|s| (s.result == 0) == wa.is_none()), "winner {wa:?}");
    }
}
