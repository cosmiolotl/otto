//! The tuner: CMA-ES over the flat [`Weights`] vector in normalised space (each entry mapped from its
//! [`Weights::BOUNDS`] to `[0, 1]`), fitness = score against a **mixed pool**.
//!
//! Per generation:
//! 1. λ candidates from CMA-ES, reflected into the unit cube.
//! 2. The pool: the fixed opponents (`fixed`, e.g. untuned default, `steady`, `greedy-app`), the most recent
//!    `pool_recent` checkpoints (the mean is checkpointed every `checkpoint_every` generations), and the
//!    current mean ("current best") if `include_mean`.
//! 3. Every candidate plays `pairs` seed pairs (both seatings) against every pool member. The seeds depend
//!    on (run seed, generation, opponent, pair) only, so all candidates of a generation see the same games
//!    (common random numbers). Fitness = −(mean score per game, draws ½).
//! 4. `tell`, then write `state.json` (CMA-ES state + counters, atomically), `mean.json`, `log.jsonl`.
//!
//! Every `eval_every` generations the mean plays `eval_pairs` fixed seed pairs against each fixed
//! opponent (`eval.jsonl`, comparable across generations). Every `oracle_every` generations one recorded
//! TL game of the mean is written as an oracle request (`oracle/`) for the M4 spot check.
//!
//! Stops after `generations` generations or `max_games` games, or when the file `STOP` exists in the
//! output directory (checked after each generation). Rerunning with the same output directory resumes.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tetris_bot::{BotSpec, SearchOptions, Weights, weights};
use tetris_sim::replay::Request;

use crate::cmaes::{Cma, default_lambda, reflect};
use crate::conditions::Conditions;
use crate::game::play_game;
use crate::schedule::{Job, PairResult, derive_seed, run};
use crate::stats::pair_score;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TuneConfig {
    /// Output directory (state, logs, checkpoints). Relative paths are relative to the working directory.
    pub out: String,
    pub seed: u64,
    /// Starting point: preset name or weights file.
    pub start: String,
    pub generations: u64,
    /// Stop once this many games have been played in total (0 = no limit).
    pub max_games: u64,
    /// λ (0 = CMA-ES default `4 + ⌊3 ln n⌋`).
    pub population: usize,
    /// Initial step size in normalised space (each weight's bounds span 1).
    pub sigma0: f64,
    /// Seed pairs per candidate per pool member per generation.
    pub pairs: usize,
    /// Beam width for every bot in tuning games.
    pub width: usize,
    /// Fixed pool members: presets or weights files.
    pub fixed: Vec<String>,
    pub include_mean: bool,
    pub checkpoint_every: u64,
    pub pool_recent: usize,
    pub eval_every: u64,
    pub eval_pairs: usize,
    pub oracle_every: u64,
    /// Rayon threads (0 = all cores).
    pub threads: usize,
    pub conditions: Conditions,
}

impl Default for TuneConfig {
    fn default() -> Self {
        TuneConfig {
            out: "runs/tune".into(),
            seed: 1,
            start: "default".into(),
            generations: 100,
            max_games: 0,
            population: 0,
            sigma0: 0.12,
            pairs: 4,
            width: 32,
            fixed: vec!["default".into(), "steady".into(), "greedy-app".into()],
            include_mean: true,
            checkpoint_every: 10,
            pool_recent: 3,
            eval_every: 10,
            eval_pairs: 50,
            oracle_every: 50,
            threads: 0,
            conditions: Conditions::default(),
        }
    }
}

impl TuneConfig {
    pub fn load(path: &Path) -> Result<TuneConfig, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn lambda(&self) -> usize {
        if self.population == 0 { default_lambda(Weights::LEN) } else { self.population }
    }

    fn opts(&self) -> SearchOptions {
        SearchOptions { width: self.width, ..Default::default() }
    }
}

/// Everything needed to resume.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneState {
    pub cma: Cma,
    pub games: u64,
    pub elapsed_s: f64,
    /// Checkpoint files (relative to the output directory), oldest first.
    pub pool: Vec<String>,
    /// Identity of the run (resuming with a different one is refused).
    pub seed: u64,
    pub start: String,
}

/// Normalised `[0, 1]` coordinates of `w` (clamped into the bounds).
pub fn to_unit(w: &Weights) -> Vec<f64> {
    w.to_vec()
        .iter()
        .zip(Weights::BOUNDS)
        .map(|(&x, &(lo, hi))| ((x - lo) as f64 / (hi - lo) as f64).clamp(0.0, 1.0))
        .collect()
}

/// `base` with its vector set from unit coordinates (reflected into `[0, 1]` first).
pub fn from_unit(base: &Weights, u: &[f64], name: String) -> Weights {
    let v: Vec<f32> = u.iter().zip(Weights::BOUNDS).map(|(&x, &(lo, hi))| lo + (hi - lo) * reflect(x) as f32).collect();
    let mut w = base.clone();
    w.set_from_vec(&v);
    w.name = name;
    w
}

#[derive(Serialize)]
struct GenLog<'a> {
    generation: u64,
    games: u64,
    wall_s: f64,
    games_per_s: f64,
    /// Placements per game (both players) and the share of games that hit the piece limit.
    pieces_per_game: f64,
    draw_rate: f64,
    sigma: f64,
    axis_sd_max: f64,
    axis_sd_min: f64,
    fitness_best: f64,
    fitness_median: f64,
    fitness_mean: f64,
    /// Mean candidate score against each pool member.
    vs: Vec<(&'a str, f64)>,
}

#[derive(Serialize)]
struct EvalLog {
    generation: u64,
    games: u64,
    vs: Vec<(String, f64, f64, f64)>,
}

fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

fn append(path: &Path, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")
}

fn weights_json(w: &Weights) -> String {
    serde_json::to_string_pretty(w).unwrap() + "\n"
}

/// Run (or resume) a tuning run. Progress goes to stderr; files to `cfg.out`.
pub fn tune(cfg: &TuneConfig) -> Result<TuneState, String> {
    if cfg.threads > 0 {
        // Fails harmlessly if the global pool already exists (e.g. in tests).
        let _ = rayon::ThreadPoolBuilder::new().num_threads(cfg.threads).build_global();
    }
    let out = PathBuf::from(&cfg.out);
    for d in [out.clone(), out.join("pool"), out.join("oracle")] {
        std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let io = |e: std::io::Error| e.to_string();
    let start = weights::load(&cfg.start)?;
    let fixed: Vec<Weights> = cfg.fixed.iter().map(|s| weights::load(s)).collect::<Result<_, _>>()?;
    let lambda = cfg.lambda();
    let state_path = out.join("state.json");
    let mut st = if state_path.exists() {
        let text = std::fs::read_to_string(&state_path).map_err(io)?;
        let mut st: TuneState = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", state_path.display()))?;
        if st.seed != cfg.seed || st.start != cfg.start || st.cma.lambda != lambda {
            return Err(format!(
                "{} belongs to another run (seed {}, start {}, λ {}); use a new `out`",
                state_path.display(),
                st.seed,
                st.start,
                st.cma.lambda
            ));
        }
        st.cma.decompose();
        eprintln!("resuming {} at generation {} ({} games so far)", cfg.out, st.cma.generation, st.games);
        st
    } else {
        let st = TuneState {
            cma: Cma::new(to_unit(&start), cfg.sigma0, lambda, derive_seed(&[cfg.seed, 0xc3a])),
            games: 0,
            elapsed_s: 0.0,
            pool: Vec::new(),
            seed: cfg.seed,
            start: cfg.start.clone(),
        };
        write_atomic(&out.join("tune.toml"), &toml::to_string(cfg).map_err(|e| e.to_string())?).map_err(io)?;
        st
    };
    let opts = cfg.opts();
    let spec = |w: Weights| BotSpec::hand(w, opts);
    let fixed_specs: Vec<BotSpec> = fixed.iter().cloned().map(spec).collect();
    eprintln!(
        "tune: n = {}, λ = {lambda}, pool = {} fixed + {} recent{}, {} pairs per candidate per opponent, width {}",
        Weights::LEN,
        fixed.len(),
        cfg.pool_recent,
        if cfg.include_mean { " + mean" } else { "" },
        cfg.pairs,
        cfg.width
    );
    loop {
        let g = st.cma.generation;
        if g >= cfg.generations || (cfg.max_games > 0 && st.games >= cfg.max_games) {
            break;
        }
        if out.join("STOP").exists() {
            eprintln!("STOP file found; stopping at generation {g}");
            break;
        }
        let t0 = Instant::now();
        // Pool for this generation.
        let mut pool: Vec<BotSpec> = fixed_specs.clone();
        for f in st.pool.iter().rev().take(cfg.pool_recent).rev() {
            pool.push(spec(weights::load(out.join(f).to_str().unwrap())?));
        }
        if cfg.include_mean {
            let mean_w = from_unit(&start, &st.cma.mean, format!("mean-{g:04}"));
            pool.push(BotSpec { name: "mean".into(), ..spec(mean_w) });
        }
        // Candidates and their games.
        let xs: Vec<Vec<f64>> = st.cma.ask().into_iter().map(|x| x.into_iter().map(reflect).collect()).collect();
        let cands: Vec<BotSpec> =
            xs.iter().enumerate().map(|(k, x)| spec(from_unit(&start, x, format!("cand-{g:04}-{k:02}")))).collect();
        let mut jobs = Vec::with_capacity(lambda * pool.len() * cfg.pairs);
        for (c, cand) in cands.iter().enumerate() {
            for (o, opp) in pool.iter().enumerate() {
                for k in 0..cfg.pairs {
                    let seed = derive_seed(&[cfg.seed, g, o as u64, k as u64]);
                    let draw = cfg.conditions.draw(seed);
                    jobs.push(Job::new((c, cand), (lambda + o, opp), &draw, cfg.conditions.max_pieces, false));
                }
            }
        }
        let res = run(&jobs);
        st.games += 2 * res.len() as u64;
        let mut score = vec![0.0; lambda];
        let mut vs = vec![0.0; pool.len()];
        let (mut pieces, mut draws) = (0u64, 0u64);
        for r in &res {
            score[r.a] += r.a_total();
            vs[r.b - lambda] += r.a_total();
            for g in &r.games {
                pieces += (g.stats[0].pieces + g.stats[1].pieces) as u64;
                draws += g.winner.is_none() as u64;
            }
        }
        let n_games = (2 * res.len()) as f64;
        let per_cand = (2 * pool.len() * cfg.pairs) as f64;
        let fitness: Vec<f64> = score.iter().map(|s| -s / per_cand).collect();
        st.cma.tell(&xs, &fitness);
        let wall = t0.elapsed().as_secs_f64();
        st.elapsed_s += wall;
        // Logs and checkpoints.
        let mut sorted = fitness.clone();
        sorted.sort_by(f64::total_cmp);
        let (hi, lo) = st.cma.axis_sd();
        let log = GenLog {
            generation: st.cma.generation,
            games: st.games,
            wall_s: wall,
            games_per_s: n_games / wall,
            pieces_per_game: pieces as f64 / n_games,
            draw_rate: draws as f64 / n_games,
            sigma: st.cma.sigma,
            axis_sd_max: hi,
            axis_sd_min: lo,
            fitness_best: -sorted[0],
            fitness_median: -sorted[lambda / 2],
            fitness_mean: -fitness.iter().sum::<f64>() / lambda as f64,
            vs: pool
                .iter()
                .map(|b| b.name.as_str())
                .zip(vs.iter().map(|v| v / (2 * lambda * cfg.pairs) as f64))
                .collect(),
        };
        append(&out.join("log.jsonl"), &serde_json::to_string(&log).unwrap()).map_err(io)?;
        eprintln!(
            "gen {:4}: {} games ({:.1}/s, {:.0} pieces, {:.0}% draws), score best {:.3} median {:.3}, σ {:.4}, \
             total {} games, {:.2} h",
            log.generation,
            2 * res.len(),
            log.games_per_s,
            log.pieces_per_game,
            100.0 * log.draw_rate,
            log.fitness_best,
            log.fitness_median,
            log.sigma,
            st.games,
            st.elapsed_s / 3600.0
        );
        let g1 = st.cma.generation;
        let mean_w = from_unit(&start, &st.cma.mean, format!("mean-{g1:04}"));
        write_atomic(&out.join("mean.json"), &weights_json(&mean_w)).map_err(io)?;
        if cfg.checkpoint_every > 0 && g1 % cfg.checkpoint_every == 0 {
            let f = format!("pool/gen-{g1:04}.json");
            write_atomic(&out.join(&f), &weights_json(&mean_w)).map_err(io)?;
            st.pool.push(f);
        }
        if cfg.eval_every > 0 && g1 % cfg.eval_every == 0 {
            let e = evaluate(&mean_w, &fixed, cfg);
            st.games += e.1;
            append(
                &out.join("eval.jsonl"),
                &serde_json::to_string(&EvalLog { generation: g1, games: st.games, vs: e.0.clone() }).unwrap(),
            )
            .map_err(io)?;
            for (name, s, lo, hi) in &e.0 {
                eprintln!("  eval mean vs {name:<12} {s:.3} [{lo:.3}, {hi:.3}]");
            }
        }
        if cfg.oracle_every > 0 && g1 % cfg.oracle_every == 0 {
            let req = oracle_game(&mean_w, &fixed[0], cfg, g1);
            write_atomic(&out.join(format!("oracle/gen-{g1:04}.json")), &serde_json::to_string(&req).unwrap())
                .map_err(io)?;
        }
        write_atomic(&state_path, &serde_json::to_string(&st).unwrap()).map_err(io)?;
    }
    Ok(st)
}

/// The mean against each fixed opponent on fixed seeds (the same every evaluation). Returns
/// `(name, score, lo, hi)` per opponent and the number of games played.
pub fn evaluate(mean: &Weights, fixed: &[Weights], cfg: &TuneConfig) -> (Vec<(String, f64, f64, f64)>, u64) {
    let opts = cfg.opts();
    let me = BotSpec::hand(mean.clone(), opts);
    let opps: Vec<BotSpec> = fixed.iter().map(|w| BotSpec::hand(w.clone(), opts)).collect();
    let mut jobs = Vec::new();
    for (o, opp) in opps.iter().enumerate() {
        for k in 0..cfg.eval_pairs {
            let draw = cfg.conditions.draw(derive_seed(&[cfg.seed, 0xe7a1, k as u64]));
            jobs.push(Job::new((0, &me), (1 + o, opp), &draw, cfg.conditions.max_pieces, false));
        }
    }
    let res = run(&jobs);
    let out = fixed
        .iter()
        .enumerate()
        .map(|(o, w)| {
            let ps: Vec<&PairResult> = res.iter().filter(|r| r.b == 1 + o).collect();
            let (s, lo, hi) = pair_score(&ps);
            (w.name.clone(), s, lo, hi)
        })
        .collect();
    (out, 2 * res.len() as u64)
}

/// One recorded TL game (2.5 PPS each, no variation) of `w` vs `opp`, as an oracle request.
pub fn oracle_game(w: &Weights, opp: &Weights, cfg: &TuneConfig, generation: u64) -> Request {
    let opts = cfg.opts();
    let cond = Conditions::tl(2.5, cfg.conditions.max_pieces);
    let draw = cond.draw(derive_seed(&[cfg.seed, 0x07ac1e, generation]));
    let job = Job::new(
        (0, &BotSpec::hand(w.clone(), opts)),
        (1, &BotSpec::hand(opp.clone(), opts)),
        &draw,
        cond.max_pieces,
        true,
    );
    play_game(&job.games[0]).request.expect("recorded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_mapping_roundtrips_default() {
        let w = Weights::default();
        let back = from_unit(&w, &to_unit(&w), w.name.clone());
        for (a, b) in w.to_vec().iter().zip(back.to_vec()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// A two-generation run, then a resume that continues to three, equals a straight three-generation run.
    #[test]
    fn resume_matches_straight_run() {
        let dir = std::env::temp_dir().join(format!("otto-tune-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = TuneConfig {
            out: dir.join("a").to_string_lossy().into_owned(),
            generations: 2,
            population: 4,
            pairs: 1,
            width: 2,
            fixed: vec!["greedy-app".into()],
            pool_recent: 1,
            checkpoint_every: 1,
            eval_every: 2,
            eval_pairs: 1,
            oracle_every: 0,
            conditions: Conditions { max_pieces: 60, ..Default::default() },
            ..Default::default()
        };
        tune(&base).unwrap();
        let resumed = tune(&TuneConfig { generations: 3, ..base.clone() }).unwrap();
        let straight =
            tune(&TuneConfig { generations: 3, out: dir.join("b").to_string_lossy().into_owned(), ..base.clone() })
                .unwrap();
        assert_eq!(resumed.cma, straight.cma);
        assert_eq!(resumed.games, straight.games);
        assert_eq!(resumed.pool, straight.pool);
        let other = TuneConfig { seed: 2, ..base };
        assert!(tune(&other).is_err(), "resuming with another seed must be refused");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
