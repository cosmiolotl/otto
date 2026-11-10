//! Training data for the value network (`sim datagen`, `docs/NNUE.md`).
//!
//! Plays self-play games between bots drawn from a roster and records sampled decision points: both
//! players' `rows-v1` features (from the **real** states; the features read only visible information),
//! the layer-stack buckets, the teacher's search score and static evaluation, and, once the game is over,
//! its result from the side to move's point of view. Games are deterministic given the seed; batches of
//! games run in parallel and are written in game order, so a data set is reproducible.
//!
//! With `siblings = K > 0`, each recorded decision is followed by up to K **sibling** records: one-ply
//! children of the decision (placements of the current or the held piece), valued by the teacher as the
//! search would value them (`eval(child, path of one lock)`) and featurised as the search's [`NnueEval`]
//! sees them (hypothetical root, opponent with the lines this lock sent). The trainer ranks siblings of
//! one decision against each other. They are the teacher's top quarter, the placement the bot chose, and
//! random others.
//!
//! [`NnueEval`]: tetris_bot::NnueEval

pub mod format;

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use tetris_bot::nnue::features::{extract_static, pending_feature};
use tetris_bot::nnue::{Features, bucket, extract};
use tetris_bot::{BotSpec, Evaluator, GEN_MODE, HandEval, Observation, PathAcc, Weights};
use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{Placement, PlayerState, SplitMix};

use crate::conditions::Conditions;
use crate::game::play_game_with;
use crate::schedule::derive_seed;
use format::{CHOSEN, DataWriter, Sample};

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
    /// Sibling records per recorded decision (0: decisions only, as in data format 1).
    pub siblings: usize,
    /// Teacher for every static score (decisions and siblings); `None`: each seat's own weights.
    pub teacher: Option<Weights>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DatagenTotals {
    pub games: u64,
    /// Records, siblings included.
    pub samples: u64,
    /// Decision records.
    pub decisions: u64,
    pub wins: [u64; 2],
    pub draws: u64,
    pub pieces: u64,
}

/// The static evaluation recorded as the teacher signal: `cfg.teacher`, else the seat's own hand-written
/// weights, or the default weights for a network bot.
fn teacher(cfg: &DatagenConfig, spec: &BotSpec) -> HandEval {
    HandEval::new(cfg.teacher.clone().or_else(|| spec.weights().cloned()).unwrap_or_default())
}

/// One-ply children of `root` with their teacher scores: (score, placement, child, lines sent).
fn children(
    root: &PlayerState,
    teacher: &HandEval,
    buf: &mut Vec<Placement>,
    out: &mut Vec<(f32, Placement, PlayerState, u32)>,
) {
    buf.clear();
    out.clear();
    movegen::generate(root, GenOptions::new(GEN_MODE, true), buf);
    for &pl in buf.iter() {
        let mut c = root.clone();
        let b2b = c.chain.b2b;
        let lock = c.apply_placement(pl);
        if lock.topout {
            continue;
        }
        let mut acc = PathAcc::default();
        acc.push(&lock, b2b, teacher.discount());
        out.push((teacher.eval(&c, &acc), pl, c, acc.sent_lines));
    }
}

/// Which children become siblings: all if there are at most `k`, else the teacher's top `k / 4` (at least
/// one), the chosen placement, and uniformly random others. Returns indices into `kids`.
fn pick_siblings(
    kids: &[(f32, Placement, PlayerState, u32)],
    chosen: Placement,
    k: usize,
    rng: &mut SplitMix,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..kids.len()).collect();
    if kids.len() <= k {
        return order;
    }
    order.sort_by(|&a, &b| kids[b].0.total_cmp(&kids[a].0));
    let top = (k / 4).max(1);
    let mut pick: Vec<usize> = order[..top].to_vec();
    let mut rest: Vec<usize> = order[top..].to_vec();
    if let Some(i) = rest.iter().position(|&i| kids[i].1 == chosen) {
        pick.push(rest.swap_remove(i));
    }
    while pick.len() < k {
        let j = rng.below(rest.len() as u64) as usize;
        pick.push(rest.swap_remove(j));
    }
    pick
}

/// One game's samples (results filled in), its winner and its total placements.
pub fn play_and_sample(cfg: &DatagenConfig, game: usize) -> (Vec<Sample>, Option<usize>, u32) {
    let mut rng = SplitMix(derive_seed(&[cfg.seed, game as u64, 0xda7a]));
    let n = cfg.bots.len() as u64;
    let seats = [rng.below(n) as usize, rng.below(n) as usize];
    let draw = cfg.cond.draw_from(&mut rng);
    let fpp = draw.fpp;
    let spec = draw.game(&cfg.bots[seats[0]], &cfg.bots[seats[1]], false, false);
    let mut teachers = [teacher(cfg, &spec.bots[0]), teacher(cfg, &spec.bots[1])];
    let mut out: Vec<(usize, Sample)> = Vec::new();
    let (mut buf, mut kids) = (Vec::new(), Vec::new());
    let g = play_game_with(&spec, |m, p, d| {
        if rng.below(cfg.every.max(1) as u64) != 0 {
            return;
        }
        let (real, opp) = (&m.players[p], &m.players[1 - p]);
        let obs = Observation::from_match(m, p);
        let t = &mut teachers[p];
        t.begin(&obs);
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
            static_score: t.eval(real, &PathAcc::default()),
            game: game as u32,
            sib: 0,
        };
        extract(real, fpp[p], &mut s.me);
        extract(opp, fpp[1 - p], &mut s.opp);
        out.push((p, s));
        if cfg.siblings == 0 {
            return;
        }
        // As the search sees it: a hypothetical root, the opponent's static half plus the lines sent.
        let root = obs.me.hypothetical(&mut SplitMix(rng.next_u64()));
        let opp_h = obs.opp.hypothetical(&mut SplitMix(0));
        let mut opp_static = Features::default();
        extract_static(&opp_h, obs.opp_fpp, &mut opp_static);
        let opp_base = opp_h.garbage.size();
        children(&root, t, &mut buf, &mut kids);
        for (n, i) in pick_siblings(&kids, d.placement, cfg.siblings, &mut rng).into_iter().enumerate() {
            let (score, pl, child, sent) = &kids[i];
            let mut sib = Sample { bucket_me: bucket(child) as u8, search_score: *score, static_score: *score, ..s };
            sib.sib = (n as u16 + 1) | if *pl == d.placement { CHOSEN } else { 0 };
            extract(child, obs.my_fpp, &mut sib.me);
            sib.opp = opp_static;
            sib.opp.idx[sib.opp.len] = pending_feature(opp_base + sent);
            sib.opp.len += 1;
            out.push((p, sib));
        }
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
        "siblings": cfg.siblings,
        "teacher": cfg.teacher.is_some(),
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
            tot.decisions += samples.iter().filter(|s| s.sib == 0).count() as u64;
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
            siblings: 0,
            teacher: None,
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

    #[test]
    fn siblings_follow_their_decision() {
        let opts = SearchOptions { width: 4, depth: 2, ..Default::default() };
        let w = weights::preset("default").unwrap();
        let cfg = DatagenConfig {
            bots: vec![BotSpec::hand(w.clone(), opts)],
            games: 1,
            seed: 7,
            cond: Conditions { max_pieces: 120, ..Default::default() },
            every: 4,
            out: PathBuf::new(),
            batch: 1,
            siblings: 8,
            teacher: Some(w),
        };
        let (a, _, _) = play_and_sample(&cfg, 0);
        let (b, _, _) = play_and_sample(&cfg, 0);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].sib, 0);
        let mut i = 0;
        while i < a.len() {
            assert_eq!(a[i].sib, 0, "record {i} starts a group");
            let mut j = i + 1;
            let mut chosen = 0;
            while j < a.len() && a[j].sib != 0 {
                assert_eq!((a[j].sib & !CHOSEN) as usize, j - i, "siblings are numbered 1..");
                assert_eq!((a[j].result, a[j].ply, a[j].game), (a[i].result, a[i].ply, a[i].game));
                assert_eq!(a[j].me.as_slice(), b[j].me.as_slice());
                chosen += (a[j].sib & CHOSEN != 0) as usize;
                j += 1;
            }
            assert!((1..=8).contains(&(j - i - 1)), "group at {i} has {} siblings", j - i - 1);
            assert!(chosen <= 1);
            i = j;
        }
    }
}
