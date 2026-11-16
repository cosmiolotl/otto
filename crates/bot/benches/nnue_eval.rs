//! `cargo bench -p tetris-bot --bench nnue_eval`: cost of one search-node evaluation (feature extraction,
//! accumulator refresh, pairwise activation, dense layers) on mid-game boards, its parts, and a parent's
//! children scored with a refresh each and with incremental accumulators.
//! Set `NNUE_NET` to an absolute checkpoint path to benchmark trained weights.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tetris_bot::nnue::{AccCache, Features, Network, Slots, extract, extract_slots};
use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{Board, IncomingGarbage, PlayerState};

fn states() -> Vec<PlayerState> {
    let boards = [
        "
        ....#.....
        .#..##....
        ##.####.##
        ####.#####
        ",
        "
        ##........
        ###...#...
        ####.###..
        #####.####
        ######.###
        G.GGGGGGGG
        GGGG.GGGGG
        GGGG.GGGGG
        ",
        "
        .....#....
        ....###...
        #...####..
        ##.######.
        ###.######
        ####.#####
        #####.####
        ######.###
        #######.##
        ########.#
        ",
    ];
    let mut v = Vec::new();
    for (i, b) in boards.iter().enumerate() {
        let mut s = PlayerState::tl(i as i64 + 1);
        s.board = Board::from_ascii(b);
        s.chain.b2b = 3 * i as i32;
        s.receive_garbage(IncomingGarbage { frame: 0, amount: 2 + i as u32, cid: 1, gameid: 2 });
        v.push(s);
    }
    v
}

fn bench(c: &mut Criterion) {
    let net = std::env::var("NNUE_NET").map_or_else(|_| Network::random(1), |path| Network::load(path).unwrap());
    let st = states();
    let opp = net.opponent_table(&st[0], 24);
    let mut g = c.benchmark_group("nnue");
    let mut i = 0;
    g.bench_function("extract", |b| {
        b.iter(|| {
            let mut f = Features::default();
            extract(black_box(&st[i % st.len()]), 24, &mut f);
            i += 1;
            black_box(f.len)
        })
    });
    let feats: Vec<Features> = st
        .iter()
        .map(|s| {
            let mut f = Features::default();
            extract(s, 24, &mut f);
            f
        })
        .collect();
    g.bench_function("refresh + pairwise", |b| {
        let mut out = [0u8; tetris_bot::nnue::net::HALF_MAX];
        b.iter(|| {
            net.side(black_box(feats[i % feats.len()].as_slice()), &mut out);
            i += 1;
            black_box(out[0])
        })
    });
    g.bench_function("extract_slots", |b| {
        b.iter(|| {
            let mut f = Slots::default();
            black_box(extract_slots(black_box(&st[i % st.len()]), 24, &[], &mut f));
            i += 1;
            black_box(f)
        })
    });
    g.bench_function("evaluate (extract + refresh + dense)", |b| {
        b.iter(|| {
            let v = net.evaluate(black_box(&st[i % st.len()]), 24, &opp, 2);
            i += 1;
            black_box(v)
        })
    });
    g.bench_function("evaluate original dense", |b| {
        b.iter(|| {
            let state = black_box(&st[i % st.len()]);
            let mut features = Features::default();
            extract(state, 24, &mut features);
            let pending = (opp.base_pending + 2).min(24) as usize;
            let raw = net.eval(features.as_slice(), &opp.sides[pending], tetris_bot::nnue::bucket(state));
            i += 1;
            black_box(Network::logit(raw))
        })
    });
    // A search's children: every placement of each state, scored per parent with a refresh per child and
    // with incremental accumulators (`evaluate_children`, as `NnueEval` in the search). Time per child.
    let gopts = GenOptions::new(tetris_bot::GEN_MODE, true);
    let families: Vec<(PlayerState, Vec<PlayerState>)> = st
        .iter()
        .map(|p| {
            let mut moves = Vec::new();
            movegen::generate(p, gopts, &mut moves);
            let kids = moves
                .iter()
                .filter_map(|&m| {
                    let mut c = p.clone();
                    (!c.apply_placement(m).topout).then_some(c)
                })
                .collect();
            (p.clone(), kids)
        })
        .collect();
    let children: usize = families.iter().map(|f| f.1.len()).sum();
    let mut out = vec![0.0; children];
    g.throughput(Throughput::Elements(children as u64));
    g.bench_function("children, refresh each (per child)", |b| {
        b.iter(|| {
            for (_, kids) in &families {
                for k in kids {
                    black_box(net.evaluate(black_box(k), 24, &opp, 2));
                }
            }
        })
    });
    let mut cache = AccCache::new();
    g.bench_function("children, incremental (per child)", |b| {
        b.iter(|| {
            let mut at = 0;
            for (parent, kids) in &families {
                let o = &mut out[at..at + kids.len()];
                net.evaluate_children(&mut cache, parent, kids, |_| 2, 24, &opp, &[], o);
                at += kids.len();
            }
            black_box(&out);
        })
    });
    g.throughput(Throughput::Elements(1));
    // The portable kernels (the path aarch64 runs), on the same machine.
    let mut portable = net.clone();
    portable.use_portable_kernels();
    let portable_opp = portable.opponent_table(&st[0], 24);
    g.bench_function("portable: refresh + pairwise", |b| {
        let mut out = [0u8; tetris_bot::nnue::net::HALF_MAX];
        b.iter(|| {
            portable.side(black_box(feats[i % feats.len()].as_slice()), &mut out);
            i += 1;
            black_box(out[0])
        })
    });
    g.bench_function("portable: evaluate", |b| {
        b.iter(|| {
            let v = portable.evaluate(black_box(&st[i % st.len()]), 24, &portable_opp, 2);
            i += 1;
            black_box(v)
        })
    });
    g.throughput(Throughput::Elements(children as u64));
    cache.clear(); // another network
    g.bench_function("portable: children, incremental (per child)", |b| {
        b.iter(|| {
            let mut at = 0;
            for (parent, kids) in &families {
                let o = &mut out[at..at + kids.len()];
                portable.evaluate_children(&mut cache, parent, kids, |_| 2, 24, &portable_opp, &[], o);
                at += kids.len();
            }
            black_box(&out);
        })
    });
    g.throughput(Throughput::Elements(1));
    g.bench_function("opponent_table (once per move)", |b| b.iter(|| black_box(net.opponent_table(&st[1], 24))));
    let wide = Network::random_l1(1, 512);
    let wide_opp = wide.opponent_table(&st[0], 24);
    g.bench_function("evaluate, L1 512", |b| {
        b.iter(|| {
            let v = wide.evaluate(black_box(&st[i % st.len()]), 24, &wide_opp, 2);
            i += 1;
            black_box(v)
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
