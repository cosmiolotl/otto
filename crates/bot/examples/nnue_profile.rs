//! Where a network bot's move goes: `cargo run --release -p tetris-bot --example nnue_profile -- NET[@WEIGHTS]
//! [moves=160] [width=96]`. `random:W` is a random `rows-v1` network of width W.
//!
//! Network self-play collects observations. For each one the search runs (alternating the order) with
//! [`NnueEval`] (incremental accumulators), with a refresh per node, and with an evaluator that replays the
//! recorded scores: the identical search at almost no evaluation cost, i.e. the search's own overhead.
//! The recorded evaluator calls (`expand` and `eval`, in search order) are then replayed through the parts
//! of the evaluation. `DIFF_BLOCKS=1` also prints the per-block feature distance of a child to its parent
//! and to its previous sibling; `NNUE_PORTABLE=1` uses the portable kernels (the aarch64 path). Pin the process to one core for stable numbers (on Windows:
//! `start /b /wait /affinity 400 nnue_profile.exe ...`).

use std::cell::RefCell;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tetris_bot::controller::Plan;
use tetris_bot::nnue::{AccCache, Features, Network, OpponentTable, Slots, extract, extract_slots, plan_features};
use tetris_bot::{Bot, Decision, Evaluator, NnueEval, Observation, PathAcc, SearchOptions, Weights, weights};
use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{Match, PlayerState, SplitMix};

/// One `eval_children` call: the parent, and each child with its `sent_lines`.
struct Call {
    parent: PlayerState,
    children: Vec<PlayerState>,
    sent: Vec<u32>,
}

/// Records the evaluator calls of a search (and the scores).
struct Recorder {
    inner: NnueEval,
    calls: RefCell<Vec<Call>>,
    scores: RefCell<Vec<f32>>,
}

impl Evaluator for Recorder {
    fn begin(&mut self, obs: &Observation) -> Plan {
        self.inner.begin(obs)
    }
    fn eval(&self, _: &PlayerState, _: &PathAcc) -> f32 {
        unreachable!("the search scores children together")
    }
    fn eval_children(&self, parent: &PlayerState, children: &[PlayerState], paths: &[PathAcc], out: &mut [f32]) {
        self.inner.eval_children(parent, children, paths, out);
        let sent = paths.iter().map(|p| p.sent_lines).collect();
        self.calls.borrow_mut().push(Call { parent: parent.clone(), children: children.to_vec(), sent });
        self.scores.borrow_mut().extend_from_slice(out);
    }
}

/// Returns the recorded scores in order: the search runs exactly as with the network.
struct Replay {
    scores: Vec<f32>,
    next: std::cell::Cell<usize>,
    plan: Plan,
}

impl Evaluator for Replay {
    fn begin(&mut self, _: &Observation) -> Plan {
        self.next.set(0);
        self.plan
    }
    fn eval(&self, _: &PlayerState, _: &PathAcc) -> f32 {
        let i = self.next.get();
        self.next.set(i + 1);
        self.scores[i]
    }
}

/// The network with a refresh per node (no incremental accumulators).
struct Refresh {
    inner: NnueEval,
    opp: Option<OpponentTable>,
    extra: Vec<u16>,
    fpp: u32,
}

impl Refresh {
    fn new(inner: NnueEval) -> Refresh {
        Refresh { inner, opp: None, extra: Vec::new(), fpp: 24 }
    }
}

impl Evaluator for Refresh {
    fn begin(&mut self, obs: &Observation) -> Plan {
        let plan = self.inner.begin(obs);
        let net = &self.inner.net;
        self.opp = Some(net.opponent_table(&obs.opp.hypothetical(&mut SplitMix(0)), obs.opp_fpp));
        self.extra = if net.layout.has_plan() { plan_features(&plan).to_vec() } else { Vec::new() };
        self.fpp = obs.my_fpp;
        plan
    }
    fn eval(&self, s: &PlayerState, path: &PathAcc) -> f32 {
        self.inner.net.evaluate_with(s, self.fpp, self.opp.as_ref().unwrap(), path.sent_lines, &self.extra)
    }
}

/// Timed repetitions of each measurement (the minimum counts: the machine may be busy).
const REPS: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().expect("NET[@WEIGHTS] [moves=160] [width=96]");
    let (net, controller): (Network, Option<Weights>) = if let Some(width) = name.strip_prefix("random:") {
        (Network::random_l1(71, width.parse().unwrap()), None)
    } else if let Some((stem, w)) = name.rsplit_once(".nnue@") {
        (Network::load(format!("{stem}.nnue")).expect("load network"), Some(weights::load(w).expect("weights")))
    } else {
        (Network::load(name).expect("load network"), None)
    };
    let mut net = net;
    if std::env::var_os("NNUE_PORTABLE").is_some() {
        net.use_portable_kernels();
    }
    let net = Arc::new(net);
    let moves: usize = args.get(1).map_or(160, |s| s.parse().unwrap());
    let width: usize = args.get(2).map_or(96, |s| s.parse().unwrap());
    let opts = SearchOptions { width, ..Default::default() };
    let new_eval = || NnueEval::with_controller(net.clone(), controller.clone());

    // Observations from network self-play: every 4th decision.
    let mut corpus = Vec::new();
    let mut bots = [Bot::with_eval(new_eval(), opts, 1), Bot::with_eval(new_eval(), opts, 2)];
    let mut game = 0;
    while corpus.len() < moves.div_ceil(4) {
        let mut m = Match::tl(3001 + game, [24, 26]);
        let mut n = 0;
        while let Some(p) = m.next_turn() {
            let obs = Observation::from_match(&m, p);
            let d = bots[p].choose(&obs).expect("legal move");
            if n % 4 == 0 {
                corpus.push(obs);
            }
            m.play(p, d.placement);
            n += 1;
            if n >= 400 || corpus.len() >= moves.div_ceil(4) {
                break;
            }
        }
        game += 1;
    }

    const LABELS: [&str; 12] = [
        "search, incremental (NnueEval)",
        "search, refresh every node",
        "search overhead (replayed scores)",
        "begin (opponent table, controller)",
        "extract",
        "extract_slots",
        "evaluate_with (refresh)",
        "evaluate_children",
        "evaluate_cached (one by one)",
        "evaluate_cached, last node only",
        "movegen (estimate)",
        "children: clone + apply (estimate)",
    ];
    let mut t = [Duration::ZERO; LABELS.len()];
    let mut stats = [0u64; 3];
    let (mut nodes, mut active) = (0u64, 0u64);
    let (mut gen_calls, mut gen_moves, mut gen_time, mut apply_time) = (0u64, 0u64, Duration::ZERO, Duration::ZERO);
    let gopts = GenOptions::new(tetris_bot::GEN_MODE, true);
    let mut buf = Vec::new();
    let timed = |t: &mut Duration, f: &mut dyn FnMut() -> Option<Decision>| {
        let s0 = Instant::now();
        let d = f();
        *t += s0.elapsed();
        d
    };
    for (index, obs) in corpus.iter().enumerate() {
        // Record the evaluator calls, then time the searches in alternating order.
        let rec = Recorder { inner: new_eval(), calls: Default::default(), scores: Default::default() };
        let mut rb = Bot::with_eval(rec, opts, 7);
        let d = rb.choose(obs);
        let (calls, scores) = (rb.eval.calls.into_inner(), rb.eval.scores.into_inner());
        let mut e = new_eval();
        let s0 = Instant::now();
        let plan = e.begin(obs);
        t[3] += s0.elapsed();
        // A fresh bot per search: a bot's root sample depends on its earlier moves.
        let mut runs: [(usize, &mut dyn FnMut() -> Option<Decision>); 3] = [
            (0, &mut || Bot::with_eval(new_eval(), opts, 7).choose(obs)),
            (1, &mut || Bot::with_eval(Refresh::new(new_eval()), opts, 7).choose(obs)),
            (2, &mut || Bot::with_eval(Replay { scores: scores.clone(), next: 0.into(), plan }, opts, 7).choose(obs)),
        ];
        let mut best = [Duration::MAX; 3];
        for r in 0..REPS {
            runs.rotate_left(1 + (index + r) % 2);
            for (i, run) in runs.iter_mut() {
                let mut time = Duration::ZERO;
                assert_eq!(timed(&mut time, *run), d);
                best[*i] = best[*i].min(time);
            }
        }
        for i in 0..3 {
            t[i] += best[i];
        }

        // Replay the calls.
        let extra: Vec<u16> = if net.layout.has_plan() { plan_features(&plan).to_vec() } else { Vec::new() };
        let opp = net.opponent_table(&obs.opp.hypothetical(&mut SplitMix(0)), obs.opp_fpp);
        let fpp = obs.my_fpp;
        let evals = || calls.iter().flat_map(|c| c.children.iter().zip(c.sent.iter().copied()));
        nodes += evals().count() as u64;
        let mut best = [Duration::MAX; 6];
        let mut out = Vec::new();
        for r in 0..REPS {
            for k in 0..6 {
                let k = (k + r) % 6;
                let s0 = Instant::now();
                match k {
                    0 => {
                        for (s, _) in evals() {
                            let mut f = Features::default();
                            extract(black_box(s), fpp, &mut f);
                            f.extend(&extra);
                            black_box(&f);
                        }
                    }
                    1 => {
                        let mut n = 0;
                        for (s, _) in evals() {
                            let mut f = Slots::default();
                            n += black_box(extract_slots(black_box(s), fpp, &extra, &mut f)) as u64;
                            black_box(&f);
                        }
                        if r == 0 {
                            active += n;
                        }
                    }
                    2 => {
                        for (s, sent) in evals() {
                            black_box(net.evaluate_with(black_box(s), fpp, &opp, sent, &extra));
                        }
                    }
                    3 => {
                        let mut cache = AccCache::new();
                        for c in &calls {
                            out.resize(c.children.len(), 0.0);
                            let sent = |i: usize| c.sent[i];
                            net.evaluate_children(
                                &mut cache,
                                &c.parent,
                                &c.children,
                                sent,
                                fpp,
                                &opp,
                                &extra,
                                &mut out,
                            );
                            black_box(&out);
                        }
                        if r == 0 {
                            for (st, v) in stats.iter_mut().zip([cache.from_parent, cache.from_last, cache.refreshed]) {
                                *st += v;
                            }
                        }
                    }
                    _ => {
                        let parents = k == 4;
                        let mut cache = AccCache::new();
                        for c in &calls {
                            if parents {
                                net.prepare_children(&mut cache, &c.parent, fpp, &extra);
                            }
                            for (s, &sent) in c.children.iter().zip(&c.sent) {
                                black_box(net.evaluate_cached(&mut cache, black_box(s), fpp, &opp, sent, &extra));
                            }
                        }
                    }
                }
                best[k] = best[k].min(s0.elapsed());
            }
        }
        for k in 0..6 {
            t[4 + k] += best[k];
        }
        // Movegen, one call per expanded parent.
        let s0 = Instant::now();
        for c in &calls {
            buf.clear();
            movegen::generate(&c.parent, gopts, &mut buf);
            gen_calls += 1;
            gen_moves += black_box(buf.len()) as u64;
        }
        gen_time += s0.elapsed();
        // Children: clone + apply for every placement of every expanded parent.
        let mut child = calls[0].parent.clone();
        let s0 = Instant::now();
        for c in &calls {
            buf.clear();
            movegen::generate(&c.parent, gopts, &mut buf);
            for &p in &buf {
                child.clone_from(&c.parent);
                black_box(child.apply_placement(p));
            }
        }
        apply_time += s0.elapsed();
    }
    t[10] = gen_time;
    t[11] = apply_time.saturating_sub(gen_time);

    let per_move = |d: Duration| d.as_secs_f64() * 1e3 / corpus.len() as f64;
    let per_node = |d: Duration| d.as_secs_f64() * 1e9 / nodes as f64;
    println!(
        "{} moves, width {width}: {:.0} nodes/move, {:.1} active features/node, {:.0} expansions/move ({:.1} moves each)",
        corpus.len(),
        nodes as f64 / corpus.len() as f64,
        active as f64 / nodes as f64,
        gen_calls as f64 / corpus.len() as f64,
        gen_moves as f64 / gen_calls as f64
    );
    for (i, label) in LABELS.iter().enumerate() {
        if i == 4 {
            println!("replayed evaluator calls, in search order:");
        }
        println!("  {label:36} {:8.2} ms/move {:7.0} ns/node", per_move(t[i]), per_node(t[i]));
    }
    let n = stats.iter().sum::<u64>() as f64;
    println!(
        "incremental accumulators: {:.1}% from the parent, {:.1}% from the last node, {:.1}% refreshed",
        100.0 * stats[0] as f64 / n,
        100.0 * stats[1] as f64 / n,
        100.0 * stats[2] as f64 / n
    );
    if std::env::var("DIFF_BLOCKS").is_ok() {
        diff_blocks(&corpus, &opts, &new_eval);
    }
}

/// Per-block feature distance of a child to its parent, and to the previous sibling (movegen order).
fn diff_blocks(corpus: &[Observation], opts: &SearchOptions, new_eval: &dyn Fn() -> NnueEval) {
    use tetris_bot::nnue::features::BLOCKS;
    let block = |i: u16| BLOCKS.iter().rposition(|b| b.offset <= i as usize).unwrap();
    let feats = |s: &PlayerState| {
        let mut f = Features::default();
        extract(s, 24, &mut f);
        f.as_slice().to_vec()
    };
    let sym = |a: &[u16], b: &[u16], out: &mut [u64; 16]| {
        for x in a.iter().filter(|x| !b.contains(x)).chain(b.iter().filter(|x| !a.contains(x))) {
            out[block(*x)] += 1;
        }
    };
    let (mut dp, mut ds) = ([0u64; 16], [0u64; 16]);
    let (mut n, mut clears) = (0u64, 0u64);
    let gopts = GenOptions::new(tetris_bot::GEN_MODE, true);
    let mut buf = Vec::new();
    for obs in corpus {
        let rec = Recorder { inner: new_eval(), calls: Default::default(), scores: Default::default() };
        let mut rb = Bot::with_eval(rec, *opts, 7);
        rb.choose(obs);
        let calls = rb.eval.calls.into_inner();
        for parent in calls.iter().map(|c| &c.parent).step_by(7) {
            let fp = feats(parent);
            buf.clear();
            movegen::generate(parent, gopts, &mut buf);
            let mut prev: Option<Vec<u16>> = None;
            for &p in &buf {
                let mut c = parent.clone();
                let out = c.apply_placement(p);
                if out.topout {
                    continue;
                }
                clears += (out.lines > 0) as u64;
                let fc = feats(&c);
                sym(&fp, &fc, &mut dp);
                if let Some(pr) = &prev {
                    sym(pr, &fc, &mut ds);
                }
                prev = Some(fc);
                n += 1;
            }
        }
    }
    println!(
        "{n} children, {:.1}% clear lines; mean feature distance per block (to the parent / to the previous sibling):",
        100.0 * clears as f64 / n as f64
    );
    for (i, b) in BLOCKS.iter().enumerate() {
        println!("  {:10} {:6.2} {:6.2}", b.name, dp[i] as f64 / n as f64, ds[i] as f64 / n as f64);
    }
    let total = |d: &[u64; 16]| d.iter().sum::<u64>() as f64 / n as f64;
    println!("  {:10} {:6.2} {:6.2}", "total", total(&dp), total(&ds));
}
