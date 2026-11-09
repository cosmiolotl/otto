//! `sim tune` and `sim tournament` (`docs/SELFPLAY.md`).

use std::path::{Path, PathBuf};

use tetris_bot::{BotSpec, SearchOptions, weights};
use tetris_selfplay::Conditions;
use tetris_selfplay::tournament::{Tournament, render, run_tournament, unique_names};
use tetris_selfplay::tune::{TuneConfig, tune as run_tune};

use crate::args::{Args, CmdResult, usage_error};

/// `sim tune --config tune/m5.toml`: run or resume (overrides for `out`, `generations`, `max_games`).
pub fn tune(args: &Args) -> CmdResult {
    let mut cfg = TuneConfig::load(Path::new(args.require("--config")?))?;
    if let Some(o) = args.get("--out") {
        cfg.out = o.to_string();
    }
    cfg.generations = args.num("--generations", cfg.generations)?;
    cfg.max_games = args.num("--max-games", cfg.max_games)?;
    let st = run_tune(&cfg)?;
    println!(
        "tune {}: {} generations, {} games, {:.2} h; mean weights in {}/mean.json",
        cfg.out,
        st.cma.generation,
        st.games,
        st.elapsed_s / 3600.0,
        cfg.out
    );
    Ok(())
}

/// The `*.json` files in `dir`, sorted.
fn json_files(dir: &str) -> Result<Vec<String>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{dir}: {e}"))?;
    let mut files: Vec<PathBuf> =
        rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    Ok(files.iter().map(|p| p.to_string_lossy().into_owned()).collect())
}

/// `sim tournament --pool runs/m5/pool --bots default,steady,greedy-app --pairs 200`: round robin.
pub fn tournament(args: &Args) -> CmdResult {
    let mut specs: Vec<String> = match args.get("--pool") {
        Some(dir) => json_files(dir)?,
        None => Vec::new(),
    };
    specs.extend(args.list("--bots"));
    let width: usize = args.num("--width", tetris_bot::DEFAULT_WIDTH)?;
    let opts = SearchOptions { width, ..Default::default() };
    let mut bots: Vec<BotSpec> = specs.iter().map(|s| BotSpec::load(s, opts)).collect::<Result<_, _>>()?;
    for s in args.list("--reserve-off") {
        bots.push(BotSpec::hand(weights::load(&s)?.reserve_off(), opts));
    }
    if bots.len() < 2 {
        return Err(usage_error("tournament needs at least two bots (--pool DIR and/or --bots a,b,...)"));
    }
    let names = unique_names(&bots);
    let max_pieces: u32 = args.num("--max-pieces", 2000)?;
    let pps = args.pair("--pps", [2.4, 2.6])?;
    let varied = args.has("--varied");
    let cond = Conditions { pps, max_pieces, tl_fraction: if varied { 0.5 } else { 1.0 }, ..Default::default() };
    let anchor = match args.get("--anchor") {
        Some(a) => {
            names.iter().position(|n| n == a).ok_or_else(|| format!("anchor {a} is not in the roster {names:?}"))?
        }
        None => names.iter().position(|n| n == "default").unwrap_or(0),
    };
    let t = Tournament {
        names,
        bots,
        pairs: args.num("--pairs", 50)?,
        seed: args.num("--seed", 1)?,
        cond,
        anchor,
        bootstrap: args.num("--bootstrap", 1000)?,
    };
    let n = t.bots.len();
    println!(
        "tournament: {n} bots, {} matchups x {} seed pairs = {} games; width {width}; {} rules; PPS {:?}; draw at {max_pieces} pieces",
        n * (n - 1) / 2,
        t.pairs,
        n * (n - 1) * t.pairs,
        if varied { "varied" } else { "TL" },
        pps
    );
    let r = run_tournament(&t);
    print!("{}", render(&r));
    if let Some(out) = args.get("--json") {
        std::fs::write(out, serde_json::to_string_pretty(&r).unwrap()).map_err(|e| format!("{out}: {e}"))?;
        println!("wrote {out}");
    }
    Ok(())
}
