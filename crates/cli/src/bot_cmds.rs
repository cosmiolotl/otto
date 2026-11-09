//! Engine tools: a head-to-head bot match, weight files.

use std::path::Path;
use std::time::Instant;

use tetris_bot::{BotSpec, Mode, SearchOptions, weights};
use tetris_selfplay::headtohead::{HeadToHead, Summary};
use tetris_selfplay::report::BotTotals;
use tetris_selfplay::{FixedSpeed, fpp_of_pps};

use crate::args::{Args, CmdResult};

/// Paired-seed bot match: `sim bot-vs-bot --a default --b greedy-app --games 1000 --seed 1`.
pub fn bot_vs_bot(args: &Args) -> CmdResult {
    let width: usize = args.num("--width", tetris_bot::DEFAULT_WIDTH)?;
    // Per bot: `--width-x` overrides `--width` (equal-time matches between evaluators of different cost);
    // `--extra-x N --k-x K` extends each leaf of the 6-ply beam greedily by N plies on K sampled
    // bag-consistent futures and averages them.
    let opts_of = |side: &str| -> Result<SearchOptions, crate::args::CliError> {
        Ok(SearchOptions {
            width: args.num(&format!("--width-{side}"), width)?,
            extra_plies: args.num(&format!("--extra-{side}"), 0)?,
            continuations: args.num(&format!("--k-{side}"), 1usize)?.max(1),
            ..Default::default()
        })
    };
    let (opts_a, opts_b) = (opts_of("a")?, opts_of("b")?);
    let a = BotSpec::load(args.get("--a").unwrap_or("default"), opts_a)?;
    let b = BotSpec::load(args.get("--b").unwrap_or("default"), opts_b)?;
    let games: usize = args.num("--games", 100)?;
    let seed: u64 = args.num("--seed", 1)?;
    let max_pieces: u32 = args.num("--max-pieces", crate::args::MAX_PIECES)?;
    let jitter: u32 = args.num("--jitter", 1)?;
    let fpp = args.pair("--pps", [2.5, 2.5])?.map(fpp_of_pps);
    let dump_dir = args.get("--dump-requests");
    let dump: usize = args.num("--dump", if dump_dir.is_some() { 3 } else { 0 })?;
    let pairs = games.div_ceil(2);
    let h = HeadToHead { a, b, pairs, seed, speed: FixedSpeed { fpp, jitter }, max_pieces, record: dump.div_ceil(2) };
    for (side, o) in [("A", &opts_a), ("B", &opts_b)] {
        if o.extra_plies > 0 {
            println!(
                "search {side}: depth {} + {} greedy plies on {} sampled futures (= {} plies)",
                o.depth,
                o.extra_plies,
                o.continuations,
                o.depth + o.extra_plies
            );
        }
    }
    println!(
        "bot-vs-bot: A = {} vs B = {}, width A {} B {}, {pairs} pairs ({} games, both seatings), seed {seed}, \
         frames/piece A {} B {} (±{jitter}/pair), draw at {max_pieces} pieces",
        h.a.name,
        h.b.name,
        opts_a.width,
        opts_b.width,
        2 * pairs,
        fpp[0],
        fpp[1]
    );
    let t = Instant::now();
    let res = h.run();
    let wall = t.elapsed().as_secs_f64();
    let s = Summary::from_pairs(&res);
    let (wr, lo, hi) = s.win_rate();
    println!("games {}: A wins {}, B wins {}, draws {}", s.a.games, s.a.wins, s.b.wins, s.a.draws);
    println!("A win rate (decisive games) {:.1}% [{:.1}%, {:.1}%] (Wilson 95%)", 100.0 * wr, 100.0 * lo, 100.0 * hi);
    println!(
        "pairs: A {} / B {} / tied {}; sign test on decisive pairs p = {:.3e}; binomial on decisive games p = {:.3e}",
        s.pairs_a,
        s.pairs_b,
        s.pairs_tied,
        s.pair_p(),
        s.game_p()
    );
    println!(
        "{:<12} {:>6} {:>9} {:>9} {:>13} {:>10} {:>8} {:>8} {:>7} {:>7}",
        "bot", "APP", "attack", "sent", "opp-cancelled", "opp-landed", "ms/move", "BUILD", "KILL", "CANCEL"
    );
    let row = |name: String, t: &BotTotals| {
        let pct = |m: Mode| format!("{:.1}%", 100.0 * t.mode_share(m));
        println!(
            "{:<12} {:>6.3} {:>9} {:>9} {:>13} {:>10} {:>8.2} {:>8} {:>7} {:>7}",
            name,
            t.app(),
            t.attack,
            t.sent,
            t.opp_cancelled,
            t.opp_landed,
            t.ms_per_move(),
            pct(Mode::Build),
            pct(Mode::Kill),
            pct(Mode::Cancel)
        );
    };
    row(format!("A {}", h.a.name), &s.a);
    row(format!("B {}", h.b.name), &s.b);
    let pieces = s.a.pieces + s.b.pieces;
    println!(
        "{:.1} pieces/game; {wall:.1} s wall ({:.0} pieces/s over all threads)",
        pieces as f64 / s.a.games.max(1) as f64,
        pieces as f64 / wall
    );
    if args.has("--log") {
        for (i, pair) in res.iter().enumerate().take(2) {
            for (g, game) in pair.games.iter().enumerate() {
                for p in 0..2 {
                    let line: String = game.mode_log[p].iter().map(|m| m.name().chars().next().unwrap()).collect();
                    println!("pair {i} game {g} seat {p} modes: {line}");
                }
            }
        }
    }
    if let Some(dir) = dump_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        let recorded = res.iter().enumerate().flat_map(|(i, pair)| {
            pair.games.iter().enumerate().filter_map(move |(g, game)| Some((i, g, game.request.as_ref()?)))
        });
        let mut n = 0;
        for (i, g, req) in recorded.take(dump) {
            let path = Path::new(dir).join(format!("bot-{i:04}-{g}.json"));
            std::fs::write(&path, serde_json::to_string(req).unwrap())
                .map_err(|e| format!("{}: {e}", path.display()))?;
            n += 1;
        }
        println!("wrote {n} oracle requests to {dir}");
    }
    Ok(())
}

/// `sim weights default -o weights/default.json`: a preset (or a partial file) as a complete JSON file.
pub fn weights(args: &Args) -> CmdResult {
    let spec = args.positional(0).ok_or_else(|| crate::args::usage_error("missing weights"))?;
    let w = weights::load(spec)?;
    let s = serde_json::to_string_pretty(&w).unwrap() + "\n";
    match args.get("-o") {
        Some(o) => std::fs::write(o, s).map_err(|e| format!("{o}: {e}"))?,
        None => print!("{s}"),
    }
    Ok(())
}
