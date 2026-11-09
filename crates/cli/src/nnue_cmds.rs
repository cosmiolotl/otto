//! Value-network tools: training data, raw network outputs (the trainer's parity check), random networks
//! (`docs/NNUE.md`).

use std::path::PathBuf;

use tetris_bot::nnue::Network;
use tetris_bot::{BotSpec, SearchOptions};
use tetris_selfplay::Conditions;
use tetris_selfplay::datagen::format::read_samples;
use tetris_selfplay::datagen::{DatagenConfig, run};

use crate::args::{Args, CmdResult, usage_error};

/// `sim datagen --bots runs/m5/pool/gen-0500.json,default --games 1000 --out data/m5-a`.
pub fn datagen(args: &Args) -> CmdResult {
    let opts = SearchOptions { width: args.num("--width", 32)?, ..Default::default() };
    let bots: Vec<BotSpec> = args.list("--bots").iter().map(|s| BotSpec::load(s, opts)).collect::<Result<_, _>>()?;
    if bots.is_empty() {
        return Err(usage_error("datagen needs --bots a,b,..."));
    }
    let out = args.require("--out")?;
    let cond = Conditions {
        pps: args.pair("--pps", [2.0, 3.0])?,
        tl_fraction: if args.has("--varied") { 0.5 } else { 1.0 },
        max_pieces: args.num("--max-pieces", 2000)?,
        ..Default::default()
    };
    let cfg = DatagenConfig {
        bots,
        games: args.num("--games", 100)?,
        seed: args.num("--seed", 1)?,
        cond,
        every: args.num("--every", 2)?,
        out: PathBuf::from(out),
        batch: args.num("--batch", 64)?,
    };
    let t = run(&cfg).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "datagen: {} games ({} / {} wins by seat, {} draws), {} samples -> {out}.{{meta,feat,json}}",
        t.games, t.wins[0], t.wins[1], t.draws, t.samples
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
