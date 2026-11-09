//! `cargo bench -p tetris-bot --bench bot_move` — criterion timing of one full bot move (observation →
//! controller → beam search → placement) at fixed beam widths, over a fixed pool of mid-game positions
//! from a seeded default-vs-default game. One iteration = one move on the next pool position.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tetris_bot::{Bot, Observation, SearchOptions, weights};
use tetris_sim::Match;

/// Observations from a seeded self-play game (default bot, width 32), every 3rd move from piece 20 on.
fn pool() -> Vec<Observation> {
    let w = weights::preset("default").unwrap();
    let opts = SearchOptions { width: 32, ..Default::default() };
    let mut bots = [Bot::new(w.clone(), opts, 1), Bot::new(w, opts, 2)];
    let mut out = Vec::new();
    for seed in [11, 22] {
        let mut m = Match::tl(seed, [24, 25]);
        let mut n = 0;
        while let Some(p) = m.next_turn() {
            let obs = Observation::from_match(&m, p);
            if n >= 20 && n % 3 == 0 {
                out.push(obs.clone());
            }
            let d = bots[p].choose(&obs).unwrap();
            m.play(p, d.placement);
            n += 1;
            if n >= 200 {
                break;
            }
        }
    }
    out
}

fn bench(c: &mut Criterion) {
    let pool = pool();
    let w = weights::preset("default").unwrap();
    let mut g = c.benchmark_group("bot_move");
    g.sample_size(20);
    for width in [16, 32, 48, 64, 96, 128] {
        let mut bot = Bot::new(w.clone(), SearchOptions { width, ..Default::default() }, 7);
        let mut i = 0;
        g.bench_function(format!("width {width}"), |b| {
            b.iter(|| {
                let d = bot.choose(black_box(&pool[i % pool.len()]));
                i += 1;
                black_box(d)
            })
        });
    }
    g.finish();

    // The value network (random weights: the cost does not depend on the values). 5 PPS = 200 ms/move.
    let net = std::sync::Arc::new(tetris_bot::nnue::Network::random(1));
    let mut g = c.benchmark_group("bot_move_nnue");
    g.sample_size(10);
    for width in [96, 192, 384, 768] {
        let eval = tetris_bot::NnueEval::new(net.clone());
        let mut bot = Bot::with_eval(eval, SearchOptions { width, ..Default::default() }, 7);
        let mut i = 0;
        g.bench_function(format!("width {width}"), |b| {
            b.iter(|| {
                let d = bot.choose(black_box(&pool[i % pool.len()]));
                i += 1;
                black_box(d)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
