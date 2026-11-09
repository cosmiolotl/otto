//! `cargo bench -p tetris-bot --bench nnue_eval`: cost of one search-node evaluation (feature extraction,
//! accumulator refresh, pairwise activation, dense layers) on mid-game boards, and its parts.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tetris_bot::nnue::{Features, Network, extract};
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
    let net = Network::random(1);
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
    g.bench_function("evaluate (extract + refresh + dense)", |b| {
        b.iter(|| {
            let v = net.evaluate(black_box(&st[i % st.len()]), 24, &opp, 2);
            i += 1;
            black_box(v)
        })
    });
    g.bench_function("opponent_table (once per move)", |b| b.iter(|| black_box(net.opponent_table(&st[1], 24))));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
