//! `cargo run --release -p tetris-bot --example inspect -- <a> <b> [seed] [width] [print_every]`:
//! play one game and print clear-type counts, b2b/surge statistics and occasional boards.

use tetris_bot::{Bot, Observation, SearchOptions, weights};
use tetris_sim::{Match, Spin};

#[derive(Default, Debug)]
struct Counts {
    singles: u32,
    doubles: u32,
    triples: u32,
    quads: u32,
    tspin: [u32; 4],
    mini: u32,
    allspin: u32,
    pc: u32,
    surges: u32,
    surge_total: u32,
    max_b2b: i32,
    breaks: u32,
    tanked: u32,
    max_height: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = weights::load(args.first().map_or("default", |s| s)).unwrap();
    let b = weights::load(args.get(1).map_or("default", |s| s)).unwrap();
    let seed: i64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let width: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);
    let every: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let opts = SearchOptions { width, ..Default::default() };
    let mut bots = [Bot::new(a, opts, 1), Bot::new(b, opts, 2)];
    let mut m = Match::tl(seed, [24, 25]);
    let mut c = [Counts::default(), Counts::default()];
    let mut n = 0;
    while let Some(p) = m.next_turn() {
        let d = bots[p].choose(&Observation::from_match(&m, p)).unwrap();
        let b2b = m.players[p].chain.b2b;
        let out = m.play(p, d.placement);
        let k = &mut c[p];
        match (out.lines, out.spin) {
            (0, _) => {}
            (l, Spin::Normal) if out.piece == tetris_sim::Piece::T => k.tspin[l as usize] += 1,
            (_, Spin::Mini) if out.piece == tetris_sim::Piece::T => k.mini += 1,
            (_, s) if s.is_spin() => k.allspin += 1,
            (1, _) => k.singles += 1,
            (2, _) => k.doubles += 1,
            (3, _) => k.triples += 1,
            _ => k.quads += 1,
        }
        k.pc += out.all_clear as u32;
        if out.surge > 0 {
            k.surges += 1;
            k.surge_total += out.surge;
        }
        if out.broke_b2b && b2b >= 0 {
            k.breaks += 1;
        }
        k.tanked += out.tanked;
        k.max_b2b = k.max_b2b.max(m.players[p].chain.b2b);
        k.max_height = k.max_height.max(m.players[p].board.top());
        n += 1;
        if every > 0 && n % every == 0 {
            println!(
                "piece {n} frame {} mode {:?} b2b {:?}",
                m.frame,
                d.mode,
                m.players.iter().map(|s| s.chain.b2b).collect::<Vec<_>>()
            );
            for pl in &m.players {
                println!("{:?} pending {}", pl.board, pl.garbage.size());
            }
        }
        if n >= 3000 {
            break;
        }
    }
    println!("winner {:?} after {n} pieces, frame {}", m.winner, m.frame);
    if std::env::var("SHOW_END").is_ok() {
        for pl in &m.players {
            println!("{:?} pending {} b2b {} mult {:.2}", pl.board, pl.garbage.size(), pl.chain.b2b, pl.multiplier());
        }
    }
    for p in 0..2 {
        let s = &m.players[p].stats;
        println!(
            "p{p} {}: pieces {} APP {:.3} sent {} cancelled {} landed {} | {:?} | modes {:?} ms/move {:.2}",
            bots[p].eval.weights.name,
            s.pieces,
            s.attack as f64 / s.pieces as f64,
            s.sent,
            s.garbage_cancelled,
            s.garbage_landed,
            c[p],
            bots[p].stats.modes,
            bots[p].stats.think_ns as f64 / 1e6 / bots[p].stats.moves as f64
        );
    }
}
