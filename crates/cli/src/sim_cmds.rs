//! Simulator tools: oracle replay and diff, request generators, perft, movegen benchmarks.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tetris_sim::movegen::fixtures::{bench_boards, midgame_states};
use tetris_sim::movegen::{self, GenOptions, Mode, Pathfinder, PerftOpts, perft_board};
use tetris_sim::replay::generate::{Style, generate, generate_movegen};
use tetris_sim::replay::{Request, diff_trace_file, replay_request_value, trace_files};
use tetris_sim::rng::park_miller_seed;
use tetris_sim::{Board, Falling, Piece, PlayerState, Spin};

use crate::args::{Args, CmdResult, usage_error};

fn write_or_print(out: Option<&str>, text: &str) -> CmdResult {
    match out {
        Some(o) => std::fs::write(o, text).map_err(|e| format!("{o}: {e}"))?,
        None => println!("{text}"),
    }
    Ok(())
}

pub fn replay(args: &Args) -> CmdResult {
    let input = args.positional(0).ok_or_else(|| usage_error("missing input"))?;
    let fail = |e: &dyn std::fmt::Display| format!("{input}: {e}");
    let text = std::fs::read_to_string(input).map_err(|e| fail(&e))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| fail(&e))?;
    let trace = replay_request_value(&v).map_err(|e| fail(&e))?;
    write_or_print(args.get("-o"), &serde_json::to_string(&trace).unwrap())
}

pub fn diff(args: &Args) -> CmdResult {
    let mut files: Vec<PathBuf> = Vec::new();
    for a in args.positionals() {
        let p = Path::new(a);
        if p.is_dir() {
            files.extend(trace_files(p).map_err(|e| format!("{a}: {e}"))?);
        } else {
            files.push(p.to_path_buf());
        }
    }
    if files.is_empty() {
        return Err(usage_error("no traces given"));
    }
    let (mut ok, mut bad, mut placements, mut matched) = (0, 0, 0usize, 0usize);
    for f in &files {
        match diff_trace_file(f) {
            Ok(r) => {
                placements += r.placements;
                matched += r.matched;
                match &r.mismatch {
                    None => ok += 1,
                    Some(m) => {
                        bad += 1;
                        println!("MISMATCH {} (placement {:?}):\n  {}", r.name, m.placement, m);
                    }
                }
            }
            Err(e) => {
                bad += 1;
                println!("ERROR {}: {e}", f.display());
            }
        }
    }
    println!("{ok}/{} traces match; {matched}/{placements} placements matched", files.len());
    if bad == 0 { Ok(()) } else { Err(format!("{bad} traces differ").into()) }
}

fn create_dir(dir: &str) -> CmdResult {
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    Ok(())
}

fn write_json(path: &Path, v: &Request) -> CmdResult {
    std::fs::write(path, serde_json::to_string(v).unwrap()).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

pub fn gen_requests(args: &Args) -> CmdResult {
    let outdir = args.positional(0).ok_or_else(|| usage_error("missing outdir"))?;
    let count: u64 = args.num("--count", 20)?;
    let seed: u64 = args.num("--seed", 1)?;
    let pieces: u32 = args.num("--pieces", 80)?;
    let styles: Vec<Style> = match args.get("--style").unwrap_or("all") {
        "all" => Style::ALL.to_vec(),
        s => vec![Style::from_name(s).ok_or_else(|| usage_error(format!("unknown style {s}")))?],
    };
    create_dir(outdir)?;
    for i in 0..count {
        let st = styles[(i % styles.len() as u64) as usize];
        let gen_seed = seed.wrapping_mul(1_000_003).wrapping_add(i);
        let req = generate(st, park_miller_seed(gen_seed.wrapping_mul(2_654_435_761)), gen_seed, pieces);
        write_json(&Path::new(outdir).join(format!("gen-{}-{:05}.json", st.name(), i)), &req)?;
    }
    println!("wrote {count} requests to {outdir}");
    Ok(())
}

/// For checking the pathfinder against the oracle: `bun src/run_trace.ts req out`, then `sim diff`.
pub fn gen_movegen(args: &Args) -> CmdResult {
    let outdir = args.positional(0).ok_or_else(|| usage_error("missing outdir"))?;
    let count: u64 = args.num("--count", 10)?;
    let seed: u64 = args.num("--seed", 1)?;
    let pieces: u32 = args.num("--pieces", 100)?;
    create_dir(outdir)?;
    let (mut placements, mut spins, mut holds) = (0, 0, 0);
    for g in 0..count {
        let (req, n) = generate_movegen(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(g), pieces);
        placements += n.placements;
        spins += n.spins;
        holds += n.holds;
        write_json(&Path::new(outdir).join(format!("mg-{g:05}.json")), &req)?;
    }
    println!("wrote {count} requests to {outdir}: {placements} placements, {spins} spins, {holds} holds");
    Ok(())
}

pub fn perft(args: &Args) -> CmdResult {
    let depth: u32 = args.num("--depth", 0)?;
    if depth == 0 {
        return Err(usage_error("missing --depth"));
    }
    let seed: i64 = args.num("--seed", 1)?;
    let mode = match args.get("--mode") {
        None => Mode::default(),
        Some(m) => Mode::from_name(m).ok_or_else(|| usage_error(format!("unknown mode {m}")))?,
    };
    let opts = PerftOpts { mode, hold: args.has("--hold"), by_cells: args.has("--by-cells") };
    let queue: Option<Vec<Piece>> = match args.get("--queue") {
        Some(q) => match q.chars().map(Piece::from_char).collect::<Option<Vec<_>>>() {
            Some(v) if !v.is_empty() => Some(v),
            _ => return Err(usage_error(format!("bad queue {q}"))),
        },
        None => None,
    };
    let state = PlayerState::tl(seed);
    let upper = |p: Piece| p.to_char().to_ascii_uppercase();
    match &queue {
        Some(q) => {
            println!("perft: empty board, cyclic queue {}, {opts:?}", q.iter().map(|&p| upper(p)).collect::<String>())
        }
        None => println!(
            "perft: PlayerState::tl({seed}), queue {}{}, {opts:?}",
            upper(state.current()),
            state.queue.iter(depth as usize).map(upper).collect::<String>()
        ),
    }
    for d in 1..=depth {
        let t = Instant::now();
        let n = match &queue {
            Some(q) => perft_board(&Board::new(), q, d, opts),
            None => movegen::perft(&state, d, opts),
        };
        let s = t.elapsed().as_secs_f64();
        println!("depth {d}: {n:>14}   {s:8.3} s   {:8.2} M leaves/s", n as f64 / s.max(1e-9) / 1e6);
    }
    Ok(())
}

pub fn bench_movegen(args: &Args) -> CmdResult {
    let budget = std::time::Duration::from_millis(args.num("--ms", 500)?);
    let mut buf = Vec::with_capacity(1024);
    println!("{:<10} {:<6} {:>10} {:>12} {:>14}", "board", "mode", "pl/call", "calls/s", "placements/s");
    for (name, board) in bench_boards() {
        for mode in [Mode::Sonic, Mode::Step] {
            let (mut calls, mut placements) = (0u64, 0u64);
            let t = Instant::now();
            while t.elapsed() < budget {
                for p in Piece::ALL {
                    buf.clear();
                    movegen::generate_from(black_box(&board), Falling::spawn(p), Spin::None, mode, false, &mut buf);
                    placements += black_box(&buf).len() as u64;
                    calls += 1;
                }
            }
            let s = t.elapsed().as_secs_f64();
            println!(
                "{name:<10} {:<6} {:>10.1} {:>12.0} {:>14.0}",
                mode.name(),
                placements as f64 / calls as f64,
                calls as f64 / s,
                placements as f64 / s
            );
        }
    }
    let pool = midgame_states(40, 120);
    let (mut step_only_states, mut extra) = (0usize, 0usize);
    let mut b2 = Vec::new();
    for s in &pool {
        buf.clear();
        b2.clear();
        movegen::generate(s, GenOptions::new(Mode::Sonic, true), &mut buf);
        movegen::generate(s, GenOptions::new(Mode::Step, true), &mut b2);
        if b2.len() > buf.len() {
            step_only_states += 1;
            extra += b2.len() - buf.len();
        }
    }
    for mode in [Mode::Sonic, Mode::Step] {
        let (mut calls, mut placements) = (0u64, 0u64);
        let t = Instant::now();
        while t.elapsed() < budget {
            for s in &pool {
                buf.clear();
                movegen::generate(black_box(s), GenOptions::new(mode, true), &mut buf);
                placements += black_box(&buf).len() as u64;
                calls += 1;
            }
        }
        let s = t.elapsed().as_secs_f64();
        println!(
            "{:<10} {:<6} {:>10.1} {:>12.0} {:>14.0}   (generate with hold: 2 pieces per call)",
            "selfplay",
            mode.name(),
            placements as f64 / calls as f64,
            calls as f64 / s,
            placements as f64 / s
        );
    }
    println!(
        "selfplay pool: {} states; Step has extra placements on {step_only_states} ({:.1}%), {extra} extra in total",
        pool.len(),
        100.0 * step_only_states as f64 / pool.len() as f64
    );
    let mut pf = Pathfinder::new();
    let mut keys = Vec::new();
    for mode in [Mode::Sonic, Mode::Step] {
        let (mut queries, mut total_keys) = (0u64, 0u64);
        let t = Instant::now();
        for s in pool.iter().step_by(4) {
            buf.clear();
            movegen::generate(s, GenOptions::new(mode, true), &mut buf);
            for p in &buf {
                keys.clear();
                assert!(pf.find(s, p, mode, &mut keys));
                total_keys += keys.len() as u64;
                queries += 1;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "pathfinder {:<6} {queries} queries, {:.1} us/query, {:.2} keys/path",
            mode.name(),
            secs * 1e6 / queries as f64,
            total_keys as f64 / queries as f64
        );
    }
    Ok(())
}
