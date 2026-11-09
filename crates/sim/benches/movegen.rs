//! `cargo bench -p tetris-sim --bench movegen` — criterion timing of the fast movegen on fixed boards
//! (`tetris_sim::movegen::fixtures`). One iteration generates the placements of all 7 pieces from their
//! spawn (7 calls); throughput is reported in placements (criterion "elements").

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tetris_sim::movegen::{self, GenOptions, Mode, fixtures::bench_boards};
use tetris_sim::{Falling, Piece, PlayerState, Spin};

fn bench(c: &mut Criterion) {
    let mut buf = Vec::with_capacity(1024);
    for (name, board) in bench_boards() {
        for mode in [Mode::Sonic, Mode::Step] {
            let mut total = 0u64;
            for p in Piece::ALL {
                buf.clear();
                movegen::generate_from(&board, Falling::spawn(p), Spin::None, mode, false, &mut buf);
                total += buf.len() as u64;
            }
            let mut g = c.benchmark_group(format!("movegen/{name}/{}", mode.name()));
            g.throughput(Throughput::Elements(total));
            g.bench_function("7 pieces", |b| {
                b.iter(|| {
                    for p in Piece::ALL {
                        buf.clear();
                        movegen::generate_from(black_box(&board), Falling::spawn(p), Spin::None, mode, false, &mut buf);
                        black_box(&buf);
                    }
                })
            });
            g.finish();
        }
    }
    // Full state API with the hold branch, mid-game board, T current + I in hold.
    let (_, board) = bench_boards().into_iter().find(|(n, _)| *n == "midgame").unwrap();
    let mut s = PlayerState::tl(1);
    s.board = board;
    s.falling = s.spawn_falling(Piece::T).0;
    s.hold = Some(Piece::I);
    for mode in [Mode::Sonic, Mode::Step] {
        buf.clear();
        movegen::generate(&s, GenOptions::new(mode, true), &mut buf);
        let mut g = c.benchmark_group(format!("movegen/midgame-hold/{}", mode.name()));
        g.throughput(Throughput::Elements(buf.len() as u64));
        g.bench_function("T+hold I", |b| {
            b.iter(|| {
                buf.clear();
                movegen::generate(black_box(&s), GenOptions::new(mode, true), &mut buf);
                black_box(&buf);
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
