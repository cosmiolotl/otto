//! Per-child cost breakdown of the search on self-play states.
use std::hint::black_box;
use std::time::Instant;

use tetris_bot::eval::board::board_features;
use tetris_bot::eval::{Evaluator, HandEval, PathAcc};
use tetris_bot::search::state_hash;
use tetris_bot::{Bot, Observation, SearchOptions, weights};
use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{Match, PlayerState};

fn main() {
    let w = weights::preset("default").unwrap();
    let opts = SearchOptions { width: 16, ..Default::default() };
    let mut bots = [Bot::new(w.clone(), opts, 1), Bot::new(w.clone(), opts, 2)];
    let mut pool: Vec<PlayerState> = Vec::new();
    let mut m = Match::tl(5, [24, 25]);
    let mut n = 0;
    while let Some(p) = m.next_turn() {
        let d = bots[p].choose(&Observation::from_match(&m, p)).unwrap();
        m.play(p, d.placement);
        pool.push(m.players[p].clone());
        n += 1;
        if n > 600 {
            break;
        }
    }
    let eval = HandEval::new(w.clone());
    let mut buf = Vec::new();
    let mut children = Vec::new();
    for s in &pool {
        buf.clear();
        movegen::generate(s, GenOptions::new(Default::default(), true), &mut buf);
        for &p in &buf {
            children.push((s.clone(), p));
        }
    }
    println!(
        "{} states, {} children ({:.1}/state)",
        pool.len(),
        children.len(),
        children.len() as f64 / pool.len() as f64
    );
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        for s in &pool {
            buf.clear();
            movegen::generate(black_box(s), GenOptions::new(Default::default(), true), &mut buf);
            black_box(&buf);
        }
    }
    let per = t.elapsed().as_nanos() as f64 / (reps * pool.len()) as f64;
    println!("movegen: {per:.0} ns/call");
    let mut scratch = pool[0].clone();
    let mut applied = Vec::new();
    let t = Instant::now();
    for r in 0..reps {
        for (s, p) in &children {
            scratch.clone_from(s);
            let out = scratch.apply_placement(*p);
            black_box(out);
            if r == 0 {
                applied.push(scratch.clone());
            }
        }
    }
    let per = t.elapsed().as_nanos() as f64 / (reps * children.len()) as f64;
    println!("clone+apply: {per:.1} ns");
    let t = Instant::now();
    for _ in 0..reps {
        for s in &applied {
            scratch.clone_from(s);
            scratch.tick_n(24);
            black_box(&scratch);
        }
    }
    println!("clone+tick_n: {:.1} ns", t.elapsed().as_nanos() as f64 / (reps * applied.len()) as f64);
    let t = Instant::now();
    for _ in 0..reps {
        for s in &applied {
            black_box(board_features(black_box(&s.board)));
        }
    }
    println!("board_features: {:.1} ns", t.elapsed().as_nanos() as f64 / (reps * applied.len()) as f64);
    let acc = PathAcc::default();
    let t = Instant::now();
    for _ in 0..reps {
        for s in &applied {
            black_box(eval.eval(black_box(s), &acc));
        }
    }
    println!("eval: {:.1} ns", t.elapsed().as_nanos() as f64 / (reps * applied.len()) as f64);
    let t = Instant::now();
    for _ in 0..reps {
        for s in &applied {
            black_box(state_hash(black_box(s)));
        }
    }
    println!("hash: {:.1} ns", t.elapsed().as_nanos() as f64 / (reps * applied.len()) as f64);
    let mut bot = Bot::new(w.clone(), SearchOptions::default(), 9);
    let obs: Vec<Observation> = {
        let mut m = Match::tl(5, [24, 25]);
        let mut v = Vec::new();
        let mut n = 0;
        while let Some(p) = m.next_turn() {
            let o = Observation::from_match(&m, p);
            let d = bots[p].choose(&o).unwrap();
            if n % 4 == 0 {
                v.push(o);
            }
            m.play(p, d.placement);
            n += 1;
            if n > 600 {
                break;
            }
        }
        v
    };
    let t = Instant::now();
    let mut nodes = 0;
    for o in &obs {
        nodes += bot.choose(o).unwrap().nodes;
    }
    let el = t.elapsed().as_nanos() as f64;
    println!(
        "full move (width {}): {:.2} ms, {:.0} children/move, {:.0} ns/child all-in",
        SearchOptions::default().width,
        el / 1e6 / obs.len() as f64,
        nodes as f64 / obs.len() as f64,
        el / nodes as f64
    );
}
