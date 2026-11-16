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
//! random others; with `keep_clears`, every placement that clears lines (every attack, every T-spin clear)
//! comes first, up to half of K, since such placements are a few percent of all children and a random draw
//! rarely includes them.
//!
//! With `sibling_search = Some((depth, width))`, every picked sibling's `search_score` is instead the
//! teacher's **search** value of that child: [`Search::run_forced`] from the decision's hypothetical root
//! with the first placement forced to the sibling, with the teacher evaluator as `begin`-ed for this
//! observation (so the controller plan is the teacher bot's), `depth` plies, beam `width`, no sampled
//! tails (`extra_plies` 0). That is the `Decision::score` the teacher would report had its search played
//! this move. The decision record's `search_score` becomes the teacher's unforced search from the same root
//! with the same options (not the playing bot's own score: the roster mixes bots), so decisions and
//! siblings share one scale. `static_score` is unchanged, and so is the sibling selection (it ranks by the
//! static score): for the same seed the records equal the run without `sibling_search` except for the
//! search scores. Exception: with `siblings = 0`, the hypothetical root is drawn from the game's RNG for the
//! decision's search alone, which shifts which decisions are sampled.
//!
//! **Teacher.** `teacher` is any bot spec ([`Teacher`]): hand-written weights, or a network (with optional
//! controller weights, `NET.nnue@WEIGHTS`). Its evaluator is `begin`-ed for every recorded decision with
//! the side to move's observation, and then scores everything above: the decision's static score is its
//! `eval(real state, empty path)`, a sibling's is `eval(child, path of the one lock)` (for a network: the
//! logit of the child against the opponent table prepared by `begin`, plus the lines this lock sent; that
//! is the value the network's search gives the node), and the search labels are the teacher's search with
//! that evaluator. A network teacher's scores are therefore logits, a hand teacher's are points; `STEM.json`
//! records which (`teacher_kind`), and the trainer calibrates each kind separately. Without a teacher each
//! seat's own hand-written weights teach (the default weights for a network seat).
//!
//! [`NnueEval`]: tetris_bot::NnueEval

pub mod format;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use tetris_bot::nnue::features::{extract_static, pending_feature};
use tetris_bot::nnue::{Features, Layout, bucket, extract, push_plan};
use tetris_bot::{
    AnyEval, BotSpec, EvalSpec, Evaluator, GEN_MODE, Observation, PathAcc, Search, SearchOptions, Weights,
};
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
    /// Teacher for every score (decisions and siblings); `None`: each seat's own hand-written weights.
    pub teacher: Option<Teacher>,
    /// Siblings: always include the line-clearing children (up to `siblings / 2`).
    pub keep_clears: bool,
    /// `(depth, width)`: label siblings and decisions with the teacher's search value instead of the
    /// static one / the playing bot's score (module docs).
    pub sibling_search: Option<(usize, usize)>,
    /// Layout `rows-v1-plan`: the side to move's features (decisions and siblings) end with the teacher's
    /// controller plan for the move (`--plan-features`; the opponent's never do).
    pub plan_features: bool,
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

/// The teacher of `sim datagen --teacher SPEC` (module docs): any evaluator a bot spec can name.
#[derive(Clone, Debug)]
pub struct Teacher {
    /// The spec as given (recorded in `STEM.json`).
    pub spec: String,
    pub eval: EvalSpec,
}

impl Teacher {
    /// A bot spec as [`BotSpec::load`] reads it: weights or a preset, `NET.nnue`, or `NET.nnue@WEIGHTS`
    /// (required for a `rows-v1-plan` network).
    pub fn load(spec: &str) -> Result<Teacher, String> {
        Ok(Teacher { spec: spec.to_string(), eval: BotSpec::load(spec, SearchOptions::default())?.eval })
    }

    pub fn hand(weights: Weights) -> Teacher {
        Teacher { spec: weights.name.clone(), eval: EvalSpec::Hand(weights) }
    }

    /// `"hand"` (scores in hand-eval points) or `"nnue"` (logits).
    pub fn kind(&self) -> &'static str {
        match self.eval {
            EvalSpec::Hand(_) => "hand",
            EvalSpec::Nnue(..) => "nnue",
        }
    }
}

/// The evaluator recorded as the teacher signal: `cfg.teacher`, else the seat's own hand-written weights,
/// or the default weights for a network bot.
fn teacher(cfg: &DatagenConfig, spec: &BotSpec) -> AnyEval {
    match &cfg.teacher {
        Some(t) => t.eval.make_eval(),
        None => EvalSpec::Hand(spec.weights().cloned().unwrap_or_default()).make_eval(),
    }
}

/// Thread time spent in whole games and in labelling their recorded decisions (teacher `begin`, scores,
/// searches, features), in ns: the progress line's labelling share. Both parts of one run run under the
/// same conditions, so the share sizes a teacher setting even where CPU seconds are noisy.
static GAME_NS: AtomicU64 = AtomicU64::new(0);
static LABEL_NS: AtomicU64 = AtomicU64::new(0);

/// Adds the time from its creation to its drop to a counter.
struct Timer<'a>(Instant, &'a AtomicU64);

impl Drop for Timer<'_> {
    fn drop(&mut self) {
        self.1.fetch_add(self.0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// A one-ply child of a decision.
struct Child {
    score: f32,
    placement: Placement,
    state: PlayerState,
    /// Lines sent by the lock (the opponent's pending grows by this much).
    sent: u32,
    clears: bool,
}

/// One-ply children of `root` with their teacher scores.
fn children<E: Evaluator>(root: &PlayerState, teacher: &E, buf: &mut Vec<Placement>, out: &mut Vec<Child>) {
    buf.clear();
    out.clear();
    movegen::generate(root, GenOptions::new(GEN_MODE, true), buf);
    // As the search does before valuing a node's children (a hint; the scores don't depend on it).
    teacher.expand(root);
    for &pl in buf.iter() {
        let mut c = root.clone();
        let b2b = c.chain.b2b;
        let lock = c.apply_placement(pl);
        if lock.topout {
            continue;
        }
        let mut acc = PathAcc::default();
        acc.push(&lock, b2b, teacher.discount());
        out.push(Child {
            score: teacher.eval(&c, &acc),
            placement: pl,
            state: c,
            sent: acc.sent_lines,
            clears: lock.lines > 0,
        });
    }
}

/// Which children become siblings: all if there are at most `k`. Else, with `keep_clears`, the
/// line-clearing children first (the teacher's best of them if there are more than `k / 2`); then the
/// teacher's top `k / 4` (at least one) of the rest, the chosen placement, and uniformly random others.
/// Returns indices into `kids`.
fn pick_siblings(kids: &[Child], chosen: Placement, k: usize, keep_clears: bool, rng: &mut SplitMix) -> Vec<usize> {
    let mut order: Vec<usize> = (0..kids.len()).collect();
    if kids.len() <= k {
        return order;
    }
    order.sort_by(|&a, &b| kids[b].score.total_cmp(&kids[a].score));
    let mut pick: Vec<usize> = Vec::with_capacity(k);
    if keep_clears {
        pick.extend(order.iter().copied().filter(|&i| kids[i].clears).take(k / 2));
        order.retain(|i| !pick.contains(i));
    }
    let top = (k / 4).max(1).min(k - pick.len());
    pick.extend_from_slice(&order[..top]);
    let mut rest: Vec<usize> = order[top..].to_vec();
    if pick.len() < k
        && let Some(i) = rest.iter().position(|&i| kids[i].placement == chosen)
    {
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
    let teacher_search =
        cfg.sibling_search.map(|(depth, width)| SearchOptions { depth, width, continuations: 1, extra_plies: 0 });
    let mut search = Search::default();
    // Only reachable if the root has no placement (then neither has the bot).
    const NO_MOVE: f32 = -1.0e6;
    let game_time = Timer(Instant::now(), &GAME_NS);
    let g = play_game_with(&spec, |m, p, d| {
        if rng.below(cfg.every.max(1) as u64) != 0 {
            return;
        }
        let _label_time = Timer(Instant::now(), &LABEL_NS);
        let (real, opp) = (&m.players[p], &m.players[1 - p]);
        let obs = Observation::from_match(m, p);
        let t = &mut teachers[p];
        // The teacher's plan for this move: the plan features of the decision and of all its siblings (the
        // search too picks the mode once, at the root).
        let plan = t.begin(&obs);
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
        if cfg.plan_features {
            push_plan(&plan, &mut s.me);
        }
        extract(opp, fpp[1 - p], &mut s.opp);
        if cfg.siblings == 0 && teacher_search.is_none() {
            out.push((p, s));
            return;
        }
        // As the search sees it: a hypothetical root, the opponent's static half plus the lines sent.
        let root = obs.me.hypothetical(&mut SplitMix(rng.next_u64()));
        if let Some(o) = &teacher_search {
            s.search_score = search.run(&root, obs.my_fpp, &*t, o, &[]).map_or(NO_MOVE, |r| r.score);
        }
        out.push((p, s));
        if cfg.siblings == 0 {
            return;
        }
        let opp_h = obs.opp.hypothetical(&mut SplitMix(0));
        let mut opp_static = Features::default();
        extract_static(&opp_h, obs.opp_fpp, &mut opp_static);
        let opp_base = opp_h.garbage.size();
        children(&root, &*t, &mut buf, &mut kids);
        for (n, i) in pick_siblings(&kids, d.placement, cfg.siblings, cfg.keep_clears, &mut rng).into_iter().enumerate()
        {
            let Child { score, placement: pl, state: child, sent, .. } = &kids[i];
            let search_score = match &teacher_search {
                Some(o) => search.run_forced(&root, obs.my_fpp, &*t, o, &[], *pl).map_or(NO_MOVE, |r| r.score),
                None => *score,
            };
            let mut sib = Sample { bucket_me: bucket(child) as u8, search_score, static_score: *score, ..s };
            sib.sib = (n as u16 + 1) | if *pl == d.placement { CHOSEN } else { 0 };
            extract(child, obs.my_fpp, &mut sib.me);
            if cfg.plan_features {
                push_plan(&plan, &mut sib.me);
            }
            sib.opp = opp_static;
            sib.opp.idx[sib.opp.len] = pending_feature(opp_base + sent);
            sib.opp.len += 1;
            out.push((p, sib));
        }
    });
    drop(game_time);
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
        // The scale of every score: hand-eval points or network logits (the trainer calibrates per kind).
        "teacher_kind": cfg.teacher.as_ref().map_or("hand", Teacher::kind),
        "teacher_spec": cfg.teacher.as_ref().map(|t| t.spec.as_str()),
        "keep_clears": cfg.keep_clears,
        "sibling_search": cfg.sibling_search.map(|(depth, width)| [depth, width]),
        "plan_features": cfg.plan_features,
        "conditions": serde_json::to_value(&cfg.cond).unwrap(),
        "bots": cfg.bots.iter().map(|b| serde_json::json!({ "name": b.name, "width": b.opts.width })).collect::<Vec<_>>(),
    });
    let layout = if cfg.plan_features { Layout::RowsV1Plan } else { Layout::RowsV1 };
    let mut w = DataWriter::create(&cfg.out, settings)?.with_layout(layout);
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
            "datagen: {}/{} games, {} samples, {:.0} pieces/game, {:.1} games/s, {:.0} samples/s, {:.1}% labelling",
            tot.games,
            cfg.games,
            tot.samples,
            tot.pieces as f64 / tot.games as f64,
            tot.games as f64 / secs,
            tot.samples as f64 / secs,
            100.0 * LABEL_NS.load(Ordering::Relaxed) as f64 / GAME_NS.load(Ordering::Relaxed).max(1) as f64
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
            keep_clears: false,
            sibling_search: None,
            plan_features: false,
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
            teacher: Some(Teacher::hand(w)),
            keep_clears: true,
            sibling_search: None,
            plan_features: false,
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

    #[test]
    fn sibling_search_only_changes_search_scores() {
        let opts = SearchOptions { width: 4, depth: 2, ..Default::default() };
        let w = weights::preset("default").unwrap();
        let base = DatagenConfig {
            bots: vec![BotSpec::hand(w.clone(), opts), BotSpec::hand(weights::preset("greedy-app").unwrap(), opts)],
            games: 1,
            seed: 9,
            cond: Conditions { max_pieces: 150, ..Default::default() },
            every: 3,
            out: PathBuf::new(),
            batch: 1,
            siblings: 8,
            teacher: Some(Teacher::hand(w)),
            keep_clears: true,
            sibling_search: None,
            plan_features: false,
        };
        let (plain, wp, pp) = play_and_sample(&base, 0);
        let deep = DatagenConfig { sibling_search: Some((3, 8)), ..base.clone() };
        let (a, wa, pa) = play_and_sample(&deep, 0);
        let (b, _, _) = play_and_sample(&deep, 0);
        let (one, _, _) = play_and_sample(&DatagenConfig { sibling_search: Some((1, 8)), ..base.clone() }, 0);
        assert_eq!((wa, pa), (wp, pp));
        assert!(plain.iter().any(|s| s.sib != 0));
        for run in [&a, &one] {
            assert_eq!(run.len(), plain.len());
            for (x, y) in run.iter().zip(&plain) {
                // Same records (same sibling selection and numbering), static scores included.
                assert_eq!((x.sib, x.game, x.ply, x.result, x.to_end), (y.sib, y.game, y.ply, y.result, y.to_end));
                assert_eq!((x.me.as_slice(), x.opp.as_slice()), (y.me.as_slice(), y.opp.as_slice()));
                assert_eq!((x.bucket_me, x.bucket_opp, x.mode), (y.bucket_me, y.bucket_opp, y.mode));
                assert_eq!(x.static_score, y.static_score);
                assert!(x.search_score.is_finite());
            }
        }
        // Deterministic.
        assert!(a.iter().zip(&b).all(|(x, y)| x.search_score == y.search_score));
        // Depth 1: a sibling's search value is its static value, and the decision's is the best child's,
        // so at least every sibling's.
        let mut i = 0;
        while i < one.len() {
            let mut j = i + 1;
            while j < one.len() && one[j].sib != 0 {
                assert_eq!(one[j].search_score, one[j].static_score);
                assert!(one[i].search_score >= one[j].search_score);
                j += 1;
            }
            i = j;
        }
        // Deeper search labels differ from static ones somewhere.
        assert!(a.iter().any(|s| s.sib != 0 && s.search_score != s.static_score));
    }

    #[test]
    fn plan_features_append_the_teachers_plan_to_my_side() {
        use tetris_bot::nnue::{N_FEATURES, N_FEATURES_PLAN};
        let opts = SearchOptions { width: 4, depth: 2, ..Default::default() };
        let w = weights::preset("default").unwrap();
        let base = DatagenConfig {
            bots: vec![BotSpec::hand(w.clone(), opts)],
            games: 1,
            seed: 7,
            cond: Conditions { max_pieces: 120, ..Default::default() },
            every: 4,
            out: PathBuf::new(),
            batch: 1,
            siblings: 6,
            teacher: Some(Teacher::hand(w)),
            keep_clears: true,
            sibling_search: None,
            plan_features: false,
        };
        let plain = play_and_sample(&base, 0).0;
        let plan = play_and_sample(&DatagenConfig { plan_features: true, ..base }, 0).0;
        assert_eq!(plain.len(), plan.len());
        let mut decision_plan: &[u16] = &[];
        for (x, y) in plain.iter().zip(&plan) {
            // Same records; my features gain the 3 plan features at the end, the opponent's none.
            assert_eq!(&y.me.as_slice()[..x.me.len], x.me.as_slice());
            assert_eq!(y.me.len, x.me.len + 3);
            assert_eq!(y.opp.as_slice(), x.opp.as_slice());
            assert!(y.opp.as_slice().iter().all(|&f| (f as usize) < N_FEATURES));
            let tail = &y.me.as_slice()[x.me.len..];
            assert!(tail.iter().all(|&f| (N_FEATURES..N_FEATURES_PLAN).contains(&(f as usize))));
            if y.sib == 0 {
                decision_plan = tail;
            } else {
                assert_eq!(tail, decision_plan, "a sibling has its decision's plan");
            }
        }
    }

    fn net_teacher(net: &std::sync::Arc<tetris_bot::nnue::Network>, controller: Option<Weights>) -> Teacher {
        Teacher { spec: "net.nnue".into(), eval: EvalSpec::Nnue(net.clone(), controller) }
    }

    #[test]
    fn network_teacher_labels_deterministically() {
        use std::sync::Arc;
        use tetris_bot::nnue::{Layout, Network};
        let opts = SearchOptions { width: 4, depth: 2, ..Default::default() };
        let net = Arc::new(Network::random_layout(3, Layout::RowsV1, 128));
        let base = DatagenConfig {
            bots: vec![
                BotSpec::hand(weights::preset("default").unwrap(), opts),
                BotSpec::nnue("net".into(), net.clone(), opts),
            ],
            games: 1,
            seed: 9,
            cond: Conditions { max_pieces: 150, ..Default::default() },
            every: 3,
            out: PathBuf::new(),
            batch: 1,
            siblings: 8,
            teacher: Some(net_teacher(&net, None)),
            keep_clears: true,
            sibling_search: Some((3, 8)),
            plan_features: false,
        };
        let (a, wa, pa) = play_and_sample(&base, 0);
        let (b, wb, pb) = play_and_sample(&base, 0);
        assert_eq!((wa, pa, a.len()), (wb, pb, b.len()));
        assert!(a.iter().any(|s| s.sib != 0));
        for (x, y) in a.iter().zip(&b) {
            assert_eq!((x.me.as_slice(), x.opp.as_slice(), x.sib), (y.me.as_slice(), y.opp.as_slice(), y.sib));
            assert_eq!((x.static_score, x.search_score), (y.static_score, y.search_score));
            assert!(x.static_score.is_finite() && x.search_score.is_finite());
        }
        // Not the hand teacher's numbers.
        let hand = play_and_sample(
            &DatagenConfig { teacher: Some(Teacher::hand(weights::preset("default").unwrap())), ..base.clone() },
            0,
        )
        .0;
        assert!(a.iter().zip(&hand).any(|(x, y)| x.sib == 0 && x.static_score != y.static_score));
        // Depth 1: a sibling's search value is the net's static value of the child, exactly as the search
        // evaluates the node; the decision's is the best child's.
        let one = play_and_sample(&DatagenConfig { sibling_search: Some((1, 8)), ..base.clone() }, 0).0;
        assert_eq!(one.len(), a.len());
        let mut i = 0;
        while i < one.len() {
            assert_eq!(one[i].static_score, a[i].static_score);
            let mut j = i + 1;
            while j < one.len() && one[j].sib != 0 {
                assert_eq!(one[j].search_score, one[j].static_score);
                assert!(one[i].search_score >= one[j].search_score);
                j += 1;
            }
            i = j;
        }
        assert!(a.iter().any(|s| s.sib != 0 && s.search_score != s.static_score));
    }

    #[test]
    fn plan_network_teacher_writes_its_plan() {
        use std::sync::Arc;
        use tetris_bot::nnue::{Layout, N_FEATURES, N_FEATURES_PLAN, Network};
        let opts = SearchOptions { width: 4, depth: 2, ..Default::default() };
        let w = weights::preset("default").unwrap();
        let net = Arc::new(Network::random_layout(4, Layout::RowsV1Plan, 128));
        let cfg = DatagenConfig {
            bots: vec![BotSpec::hand(w.clone(), opts)],
            games: 1,
            seed: 7,
            cond: Conditions { max_pieces: 120, ..Default::default() },
            every: 4,
            out: PathBuf::new(),
            batch: 1,
            siblings: 6,
            teacher: Some(net_teacher(&net, Some(w))),
            keep_clears: true,
            sibling_search: Some((2, 4)),
            plan_features: true,
        };
        let a = play_and_sample(&cfg, 0).0;
        assert!(a.iter().any(|s| s.sib != 0));
        let mut decision_plan: &[u16] = &[];
        for s in &a {
            let tail = &s.me.as_slice()[s.me.len - 3..];
            assert!(tail.iter().all(|&f| (N_FEATURES..N_FEATURES_PLAN).contains(&(f as usize))));
            if s.sib == 0 {
                decision_plan = tail;
            } else {
                assert_eq!(tail, decision_plan, "a sibling has its decision's plan");
            }
        }
        let b = play_and_sample(&cfg, 0).0;
        assert!(a.iter().zip(&b).all(|(x, y)| x.search_score == y.search_score && x.me.as_slice() == y.me.as_slice()));
    }
}
