//! `cargo run --release -p tetris-bot --example nnue_probe -- <net.nnue> [teacher weights] [games]`:
//! how a value network ranks the one-ply alternatives at real self-play positions, against the
//! hand-written evaluation (the teacher). The teacher plays the games; at every 4th decision, every
//! placement of the current piece (hold included) is scored by both evaluators.
//!
//! Prints the mean Spearman rank correlation, how often the net's best placement equals the teacher's,
//! and the average board after each one's favourite placement (holes, max height, lines cleared).

use std::sync::Arc;

use tetris_bot::eval::board::board_features;
use tetris_bot::eval::{Evaluator, HandEval, NnueEval, PathAcc};
use tetris_bot::nnue::Network;
use tetris_bot::{Bot, GEN_MODE, Observation, SearchOptions, weights};
use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{Match, SplitMix};

fn ranks(x: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        for k in i..=j {
            r[idx[k]] = (i + j) as f64 / 2.0;
        }
        i = j + 1;
    }
    r
}

fn spearman(a: &[f32], b: &[f32]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..ra.len() {
        c += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma).powi(2);
        vb += (rb[i] - mb).powi(2);
    }
    if va == 0.0 || vb == 0.0 { 0.0 } else { c / (va * vb).sqrt() }
}

#[derive(Default)]
struct Pick {
    holes: f64,
    height: f64,
    lines: f64,
    sent: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let net = Arc::new(Network::load(&args[0]).expect("load net"));
    let w = weights::load(args.get(1).map_or("default", |s| s)).unwrap();
    let games: i64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let opts = SearchOptions { width: 32, ..Default::default() };
    let gopts = GenOptions::new(GEN_MODE, true);
    let (mut hand, mut nnue) = (HandEval::new(w.clone()), NnueEval::new(net));
    let (mut n, mut rho, mut agree) = (0usize, 0.0, 0usize);
    let (mut ph, mut pn) = (Pick::default(), Pick::default());
    let mut nnue_rank_of_teacher = 0.0;
    let mut buf = Vec::new();
    for g in 0..games {
        let mut bots = [Bot::new(w.clone(), opts, 1), Bot::new(w.clone(), opts, 2)];
        let mut m = Match::tl(1000 + g, [24, 24]);
        let mut k = 0u32;
        while let Some(p) = m.next_turn() {
            let obs = Observation::from_match(&m, p);
            k += 1;
            if k % 4 == 0 {
                hand.begin(&obs);
                nnue.begin(&obs);
                let root = obs.me.hypothetical(&mut SplitMix(k as u64));
                buf.clear();
                movegen::generate(&root, gopts, &mut buf);
                let (mut hs, mut ns, mut info) = (Vec::new(), Vec::new(), Vec::new());
                for &pl in &buf {
                    let mut s = root.clone();
                    let b2b = s.chain.b2b;
                    let out = s.apply_placement(pl);
                    if out.topout {
                        continue;
                    }
                    let mut acc = PathAcc::default();
                    acc.push(&out, b2b, 1.0);
                    hs.push(hand.eval(&s, &acc));
                    ns.push(nnue.eval(&s, &acc));
                    let f = board_features(&s.board);
                    info.push((f.holes as f64, f.max_height as f64, out.lines as f64, out.sent.total() as f64));
                }
                if hs.len() >= 3 {
                    let best = |x: &[f32]| (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap();
                    let (bh, bn) = (best(&hs), best(&ns));
                    n += 1;
                    rho += spearman(&hs, &ns);
                    agree += (bh == bn) as usize;
                    // Fraction of alternatives the net scores above the teacher's favourite (0 = net agrees).
                    nnue_rank_of_teacher += ns.iter().filter(|&&v| v > ns[bh]).count() as f64 / ns.len() as f64;
                    for (pick, i) in [(&mut ph, bh), (&mut pn, bn)] {
                        pick.holes += info[i].0;
                        pick.height += info[i].1;
                        pick.lines += info[i].2;
                        pick.sent += info[i].3;
                    }
                }
            }
            let d = bots[p].choose(&obs).unwrap();
            m.play(p, d.placement);
        }
    }
    let nf = n as f64;
    println!("{n} positions from {games} games");
    println!("mean Spearman(hand, net) over one-ply alternatives: {:.3}", rho / nf);
    println!("net's best == teacher's best: {:.1}%", 100.0 * agree as f64 / nf);
    println!("alternatives the net prefers over the teacher's pick: {:.1}%", 100.0 * nnue_rank_of_teacher / nf);
    for (name, p) in [("teacher pick", &ph), ("net pick", &pn)] {
        println!(
            "{name:<13} holes {:.2}  max height {:.2}  lines {:.3}  sent {:.3}",
            p.holes / nf,
            p.height / nf,
            p.lines / nf,
            p.sent / nf
        );
    }
}
