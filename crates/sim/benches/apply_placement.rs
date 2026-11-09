//! `cargo bench -p tetris-sim` — timing for `apply_placement` (no criterion; std::time only).
//!
//! Times `clone + apply_placement` and `clone` alone over mid-game states ([`midgame_states`]).

use std::hint::black_box;
use std::time::Instant;

use tetris_sim::movegen::fixtures::midgame_states;
use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{Placement, PlayerState};

fn main() {
    // Each state paired with one of its placements (spread over the move list).
    let mut buf = Vec::new();
    let pool: Vec<(PlayerState, Placement)> = midgame_states(120, 150)
        .into_iter()
        .enumerate()
        .filter_map(|(i, s)| {
            buf.clear();
            movegen::generate(&s, GenOptions::default(), &mut buf);
            let p = *buf.get(i * 7 % buf.len().max(1))?;
            Some((s, p))
        })
        .collect();
    println!("pool: {} (state, placement) pairs", pool.len());

    let iters = 50usize;
    let n = pool.len() * iters;

    let t = Instant::now();
    for _ in 0..iters {
        for (s, _) in &pool {
            black_box(black_box(s).clone());
        }
    }
    let clone_ns = t.elapsed().as_nanos() as f64 / n as f64;

    let t = Instant::now();
    let mut lines = 0u64;
    for _ in 0..iters {
        for (s, p) in &pool {
            let mut s = black_box(s).clone();
            let out = s.apply_placement(black_box(*p));
            lines += out.lines as u64;
            black_box(&s);
        }
    }
    let apply_ns = t.elapsed().as_nanos() as f64 / n as f64;
    black_box(lines);

    println!("PlayerState size: {} bytes", std::mem::size_of::<PlayerState>());
    println!("clone:                 {clone_ns:7.1} ns");
    println!("clone + apply:         {apply_ns:7.1} ns");
    println!(
        "apply (difference):    {:7.1} ns  ({:.1} M placements/s)",
        apply_ns - clone_ns,
        1e3 / (apply_ns - clone_ns)
    );
}
