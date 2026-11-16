//! Value-network tools: training data, raw network outputs (the trainer's parity check), random networks
//! (`docs/NNUE.md`).

use std::path::PathBuf;

use tetris_bot::nnue::Network;
use tetris_bot::{BotSpec, SearchOptions};
use tetris_selfplay::datagen::format::read_samples;
use tetris_selfplay::datagen::{DatagenConfig, Teacher, run};

use crate::args::{Args, CliError, CmdResult, conditions, usage_error};

/// `--sibling-search DEPTH,WIDTH`: label siblings and decisions with the teacher's search value
/// (`tetris_selfplay::datagen` module docs).
fn sibling_search(args: &Args) -> Result<Option<(usize, usize)>, CliError> {
    let Some(s) = args.get("--sibling-search") else { return Ok(None) };
    let bad = || usage_error(format!("--sibling-search {s}: expected DEPTH,WIDTH (both at least 1)"));
    let (d, w) = s.split_once(',').ok_or_else(bad)?;
    let (d, w): (usize, usize) = (d.trim().parse().map_err(|_| bad())?, w.trim().parse().map_err(|_| bad())?);
    if d == 0 || w == 0 {
        return Err(bad());
    }
    Ok(Some((d, w)))
}

/// `sim datagen --bots runs/m5/pool/gen-0500.json,default --games 1000 --out data/m5-a`.
pub fn datagen(args: &Args) -> CmdResult {
    let opts = SearchOptions { width: args.num("--width", 32)?, ..Default::default() };
    let bots: Vec<BotSpec> = args.list("--bots").iter().map(|s| BotSpec::load(s, opts)).collect::<Result<_, _>>()?;
    if bots.is_empty() {
        return Err(usage_error("datagen needs --bots a,b,..."));
    }
    let out = args.require("--out")?;
    let cond = conditions(args, [2.0, 3.0])?;
    let cfg = DatagenConfig {
        bots,
        games: args.num("--games", 100)?,
        seed: args.num("--seed", 1)?,
        cond,
        every: args.num("--every", 2)?,
        out: PathBuf::from(out),
        batch: args.num("--batch", 64)?,
        siblings: args.num("--siblings", 0)?,
        teacher: args.get("--teacher").map(Teacher::load).transpose()?,
        keep_clears: args.has("--keep-clears"),
        sibling_search: sibling_search(args)?,
        plan_features: args.has("--plan-features"),
    };
    let t = run(&cfg).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "datagen: {} games ({} / {} wins by seat, {} draws), {} decisions, {} samples -> {out}.{{meta,feat,json}}",
        t.games, t.wins[0], t.wins[1], t.draws, t.decisions, t.samples
    );
    Ok(())
}

/// `sim nnue-eval --net X.nnue --data STEM --n 100`: one line per sample, `index bucket result raw logit`.
pub fn nnue_eval(args: &Args) -> CmdResult {
    let net = Network::load(args.require("--net")?)?;
    let data = args.require("--data")?;
    let samples = read_samples(data).map_err(|e| format!("{data}: {e}"))?;
    let n: usize = args.num("--n", samples.len())?;
    for (i, s) in samples.iter().take(n).enumerate() {
        let raw = net.eval_pair(&s.me, &s.opp, s.bucket_me as usize);
        println!("{i} {} {} {raw} {:.6}", s.bucket_me, s.result, Network::logit(raw));
    }
    Ok(())
}

/// `sim nnue-random -o X.nnue --seed 1`.
pub fn nnue_random(args: &Args) -> CmdResult {
    let out = args.require("-o")?;
    Network::random(args.num("--seed", 1)?).save(out).map_err(|e| format!("{out}: {e}"))?;
    Ok(())
}
