//! End-to-end parity and timing against the original, unprepared NNUE forward pass (a refresh per node, no
//! prepared opponent, no incremental accumulators).
//! `cargo run --release -p tetris-bot --example nnue_replay -- NET[@WEIGHTS] [games=4] [width=96]`
//! Use `random:128`, `random:256`, `random:384`, or `random:512` to exercise other widths, and
//! `random-plan:W` for a `rows-v1-plan` network (with the default controller). `@WEIGHTS` runs the release
//! controller with those weights (required for a `rows-v1-plan` network); the original evaluator runs its
//! own controller and appends the plan features itself. `NNUE_PORTABLE=1` runs the optimised evaluator on the
//! portable kernels (the path aarch64 runs) against the original on the default ones.

use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tetris_bot::controller::burst::Burst;
use tetris_bot::controller::{self, Plan};
use tetris_bot::nnue::features::{PENDING_MAX, extract_static, pending_feature};
use tetris_bot::nnue::net::{HALF_MAX, SideOut};
use tetris_bot::nnue::{Features, Layout, Network, bucket, extract, push_plan};
use tetris_bot::{Bot, Evaluator, NnueEval, Observation, PathAcc, SearchOptions, Weights, weights};
use tetris_sim::{Match, PlayerState, SplitMix};

#[path = "nnue_replay/coverage.rs"]
mod coverage;

struct OriginalEval {
    net: Arc<Network>,
    fpp: u32,
    pending: u32,
    sides: Box<[SideOut; PENDING_MAX + 1]>,
    controller: Option<Weights>,
    burst: Burst,
    plan: Plan,
}

impl OriginalEval {
    fn new(net: Arc<Network>, controller: Option<Weights>) -> Self {
        Self {
            net,
            fpp: 24,
            pending: 0,
            sides: Box::new([[0; HALF_MAX]; PENDING_MAX + 1]),
            controller,
            burst: Burst::default(),
            plan: Plan::default(),
        }
    }
}

impl Evaluator for OriginalEval {
    fn begin(&mut self, obs: &Observation) -> Plan {
        self.fpp = obs.my_fpp;
        let opponent = obs.opp.hypothetical(&mut SplitMix(0));
        self.pending = opponent.garbage.size();
        let mut features = Features::default();
        extract_static(&opponent, obs.opp_fpp, &mut features);
        let count = features.len;
        features.len += 1;
        for (pending, side) in self.sides.iter_mut().enumerate() {
            features.idx[count] = pending_feature(pending as u32);
            self.net.side(features.as_slice(), side);
        }
        self.plan = match &self.controller {
            Some(w) if w.controller => controller::decide(obs, w, &mut self.burst),
            _ => Plan::default(),
        };
        self.plan
    }

    fn eval(&self, state: &PlayerState, path: &PathAcc) -> f32 {
        let mut features = Features::default();
        extract(state, self.fpp, &mut features);
        if self.net.layout.has_plan() {
            push_plan(&self.plan, &mut features);
        }
        let pending = (self.pending + path.sent_lines).min(PENDING_MAX as u32) as usize;
        Network::logit(self.net.eval(features.as_slice(), &self.sides[pending], bucket(state)))
    }
}

struct CheckedEval {
    original: OriginalEval,
    optimized: NnueEval,
    checked: Cell<u64>,
}

impl Evaluator for CheckedEval {
    fn begin(&mut self, obs: &Observation) -> Plan {
        let expected = self.original.begin(obs);
        let actual = self.optimized.begin(obs);
        assert_eq!(actual, expected, "plan differs");
        actual
    }

    fn eval(&self, state: &PlayerState, path: &PathAcc) -> f32 {
        let expected = self.original.eval(state, path);
        let actual = self.optimized.eval(state, path);
        assert_eq!(actual.to_bits(), expected.to_bits(), "node score differs");
        self.checked.set(self.checked.get() + 1);
        actual
    }

    /// The search's path: the optimised evaluator scores a parent's children together (incremental
    /// accumulators); each score must equal the original's.
    fn eval_children(&self, parent: &PlayerState, children: &[PlayerState], paths: &[PathAcc], out: &mut [f32]) {
        self.optimized.eval_children(parent, children, paths, out);
        for ((state, path), actual) in children.iter().zip(paths).zip(out.iter()) {
            let expected = self.original.eval(state, path);
            assert_eq!(actual.to_bits(), expected.to_bits(), "node score differs");
        }
        self.checked.set(self.checked.get() + children.len() as u64);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().expect("NET[@WEIGHTS] [games=4] [width=96]");
    let (net, controller) = if let Some(width) = name.strip_prefix("random:") {
        (Network::random_l1(71, width.parse().unwrap()), None)
    } else if let Some(width) = name.strip_prefix("random-plan:") {
        (Network::random_layout(71, Layout::RowsV1Plan, width.parse().unwrap()), Some(Weights::default()))
    } else if let Some((stem, w)) = name.rsplit_once(".nnue@") {
        (Network::load(format!("{stem}.nnue")).expect("load network"), Some(weights::load(w).expect("weights")))
    } else {
        (Network::load(name).expect("load network"), None)
    };
    assert!(controller.is_some() || !net.layout.has_plan(), "a rows-v1-plan network needs NET@WEIGHTS");
    // The original evaluator keeps the default kernels; with NNUE_PORTABLE the optimised one runs the
    // portable kernels (the aarch64 path), so the two paths are checked against each other.
    let reference = Arc::new(net.clone());
    let mut net = net;
    if std::env::var_os("NNUE_PORTABLE").is_some() {
        net.use_portable_kernels();
    }
    let net = Arc::new(net);
    let games: i64 = args.get(1).map_or(4, |s| s.parse().unwrap());
    println!("{} exact evaluations across all heights and pending values", coverage::verify(&net));
    let width = args.get(2).map_or(96, |s| s.parse().unwrap());
    let mut corpus = Vec::new();
    let mut checked_nodes = 0;
    for game in 0..games {
        // Exercise both ordinary preview search and sampled continuations.
        let opts = SearchOptions {
            width,
            continuations: 1 + game as usize % 2,
            extra_plies: game as usize % 2,
            ..Default::default()
        };
        let mut original = Bot::with_eval(OriginalEval::new(reference.clone(), controller.clone()), opts, 19);
        let checked = CheckedEval {
            original: OriginalEval::new(reference.clone(), controller.clone()),
            optimized: NnueEval::with_controller(net.clone(), controller.clone()),
            checked: Cell::new(0),
        };
        let mut optimized = Bot::with_eval(checked, opts, 19);
        let mut match_state = Match::tl(2001 + game, [18 + game as u32 % 10, 28]);
        let mut moves = 0;
        while let Some(player) = match_state.next_turn() {
            let observation = Observation::from_match(&match_state, player);
            let expected = original.choose(&observation);
            let actual = optimized.choose(&observation);
            assert_eq!(actual, expected, "search differs in game {game}, move {moves}");
            if moves % 8 == 0 {
                corpus.push(observation);
            }
            let decision = actual.expect("live player has a legal move");
            match_state.play(player, decision.placement);
            moves += 1;
            if moves >= 240 {
                break;
            }
        }
        checked_nodes += optimized.eval.checked.get();
        println!("game {game}: {moves} identical decisions, {} identical node scores", optimized.eval.checked.get());
    }

    let opts = SearchOptions { width, ..Default::default() };
    let mut original = Bot::with_eval(OriginalEval::new(reference.clone(), controller.clone()), opts, 7);
    let mut optimized = Bot::with_eval(NnueEval::with_controller(net, controller), opts, 7);
    let (mut before, mut after) = (Duration::ZERO, Duration::ZERO);
    // Alternate order to reduce clock/thermal bias; keep validation outside the timed regions.
    for (index, observation) in corpus.iter().enumerate() {
        let mut run_original = || {
            let start = Instant::now();
            let decision = original.choose(observation);
            before += start.elapsed();
            decision
        };
        let mut run_optimized = || {
            let start = Instant::now();
            let decision = optimized.choose(observation);
            after += start.elapsed();
            decision
        };
        let (expected, actual) = if index % 2 == 0 {
            (run_original(), run_optimized())
        } else {
            let actual = run_optimized();
            (run_original(), actual)
        };
        assert_eq!(actual, expected);
    }
    println!(
        "{}",
        serde_json::json!({
            "network": name, "l1": original.eval.net.l1, "layout": original.eval.net.layout.name(), "width": width,
            "checked_nodes": checked_nodes, "timed_moves": corpus.len(),
            "original_ms_per_move": before.as_secs_f64() * 1000.0 / corpus.len() as f64,
            "optimized_ms_per_move": after.as_secs_f64() * 1000.0 / corpus.len() as f64,
            "speedup": before.as_secs_f64() / after.as_secs_f64(),
        })
    );
}
