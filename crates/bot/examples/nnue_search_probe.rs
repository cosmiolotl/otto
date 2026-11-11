//! `cargo run --release -p tetris-bot --example nnue_search_probe -- <net.nnue> [teacher weights] [games]`:
//! where a value network's search departs from the hand-written evaluation's (the teacher's). The teacher
//! plays the games; at every 4th decision:
//!
//! - **Depth 2.** Every two-placement line (hold included) is scored by both evaluators. The probe reports
//!   the Spearman correlation inside each first placement's children (what sibling training teaches),
//!   pooled over all lines (what a beam compares), and of the per-parent means (the offset between
//!   parents).
//! - **Search agreement.** The net's and the teacher's beam searches (width 96, depths 1, 2 and 6) choose
//!   a move from the same observation. A second teacher search with another seed gives the noise floor.

use std::sync::Arc;

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

/// Mean and count accumulator.
#[derive(Default)]
struct Mean(f64, usize);

impl Mean {
    fn add(&mut self, x: f64) {
        self.0 += x;
        self.1 += 1;
    }
    fn get(&self) -> f64 {
        self.0 / self.1.max(1) as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let net = Arc::new(Network::load(&args[0]).expect("load net"));
    let w = weights::load(args.get(1).map_or("default", |s| s)).unwrap();
    let games: i64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let gopts = GenOptions::new(GEN_MODE, true);
    let (mut hand, mut nnue) = (HandEval::new(w.clone()), NnueEval::new(net.clone()));
    let disc = hand.discount();
    let (mut within, mut pooled, mut parents, mut top_pooled) =
        (Mean::default(), Mean::default(), Mean::default(), Mean::default());
    let depths = [1usize, 2, 6];
    let mut agree_net = [0usize; 3];
    let mut agree_seed = [0usize; 3];
    let mut n_search = 0usize;
    // Depth-6 disagreements: what the teacher's move and the net's move do on the lock.
    let mut dis = [[0usize; 2]; 5]; // rows: attacks, clears, breaks B2B, T-spin, tanks garbage; cols: teacher, net
    let mut n_dis = 0usize;
    // One-ply argmax disagreements: the teacher's regret in its own points.
    let mut regret = [0usize; 4]; // < 1, 1-5, 5-20, >= 20
    let mut regret_sum = 0.0f64;
    let mut n_one = 0usize;
    // Centred residual (net - mean) - a (teacher - mean), in logits, by kind of placement.
    const A: f32 = 0.0287; // the v2 trainer's fitted logit-per-point scale
    let mut resid: [Mean; 4] = Default::default(); // no clear, clear without attack, attack, T-spin attack
    let mut abs_resid: [Mean; 4] = Default::default();
    let (mut buf1, mut buf2) = (Vec::new(), Vec::new());
    let play = SearchOptions { width: 32, ..Default::default() };
    for g in 0..games {
        let mut bots = [Bot::new(w.clone(), play, 1), Bot::new(w.clone(), play, 2)];
        let mut m = Match::tl(1000 + g, [24, 24]);
        let mut k = 0u32;
        while let Some(p) = m.next_turn() {
            let obs = Observation::from_match(&m, p);
            k += 1;
            if k % 4 == 0 {
                hand.begin(&obs);
                nnue.begin(&obs);
                let root = obs.me.hypothetical(&mut SplitMix(k as u64));
                buf1.clear();
                movegen::generate(&root, gopts, &mut buf1);
                let (mut hs, mut ns) = (Vec::new(), Vec::new());
                let (mut ph, mut pn) = (Vec::new(), Vec::new());
                for &p1 in &buf1 {
                    let mut c = root.clone();
                    let b0 = c.chain.b2b;
                    let o1 = c.apply_placement(p1);
                    if o1.topout {
                        continue;
                    }
                    let mut acc1 = PathAcc::default();
                    acc1.push(&o1, b0, disc);
                    buf2.clear();
                    movegen::generate(&c, gopts, &mut buf2);
                    let (mut h2, mut n2) = (Vec::new(), Vec::new());
                    for &p2 in &buf2 {
                        let mut gc = c.clone();
                        let b1 = gc.chain.b2b;
                        let o2 = gc.apply_placement(p2);
                        if o2.topout {
                            continue;
                        }
                        let mut acc = acc1;
                        acc.push(&o2, b1, disc);
                        h2.push(hand.eval(&gc, &acc));
                        n2.push(nnue.eval(&gc, &acc));
                    }
                    if h2.len() >= 3 {
                        within.add(spearman(&h2, &n2));
                    }
                    if !h2.is_empty() {
                        ph.push(h2.iter().sum::<f32>() / h2.len() as f32);
                        pn.push(n2.iter().sum::<f32>() / n2.len() as f32);
                    }
                    hs.extend(h2);
                    ns.extend(n2);
                }
                {
                    let mut acc_kids = Vec::new();
                    for &p1 in &buf1 {
                        let mut c = root.clone();
                        let b0 = c.chain.b2b;
                        let o1 = c.apply_placement(p1);
                        if o1.topout {
                            continue;
                        }
                        let mut acc = PathAcc::default();
                        acc.push(&o1, b0, disc);
                        let kind = if o1.raw.total() == 0 {
                            if o1.lines == 0 { 0 } else { 1 }
                        } else if o1.piece == tetris_sim::Piece::T && o1.spin.is_spin() {
                            3
                        } else {
                            2
                        };
                        acc_kids.push((hand.eval(&c, &acc), nnue.eval(&c, &acc), kind));
                    }
                    if acc_kids.len() >= 2 {
                        let nk = acc_kids.len() as f32;
                        let mh = acc_kids.iter().map(|x| x.0).sum::<f32>() / nk;
                        let mn = acc_kids.iter().map(|x| x.1).sum::<f32>() / nk;
                        for &(h, n, kind) in &acc_kids {
                            let r = ((n - mn) - A * (h - mh)) as f64;
                            resid[kind].add(r);
                            abs_resid[kind].add(r.abs());
                        }
                    }
                    if acc_kids.len() >= 2 {
                        let bt = (0..acc_kids.len()).max_by(|&a, &b| acc_kids[a].0.total_cmp(&acc_kids[b].0)).unwrap();
                        let bn = (0..acc_kids.len()).max_by(|&a, &b| acc_kids[a].1.total_cmp(&acc_kids[b].1)).unwrap();
                        n_one += 1;
                        if bt != bn {
                            let r = (acc_kids[bt].0 - acc_kids[bn].0) as f64;
                            regret_sum += r;
                            regret[if r < 1.0 {
                                0
                            } else if r < 5.0 {
                                1
                            } else if r < 20.0 {
                                2
                            } else {
                                3
                            }] += 1;
                        }
                    }
                }
                if hs.len() >= 3 && ph.len() >= 3 {
                    pooled.add(spearman(&hs, &ns));
                    parents.add(spearman(&ph, &pn));
                    // Of the teacher's top 32 lines, how many are in the net's top 32 (a width-32 beam's overlap)?
                    let top = |x: &[f32]| {
                        let mut i: Vec<usize> = (0..x.len()).collect();
                        i.sort_by(|&a, &b| x[b].total_cmp(&x[a]));
                        i.truncate(32);
                        i
                    };
                    let (th, tn) = (top(&hs), top(&ns));
                    top_pooled.add(th.iter().filter(|i| tn.contains(i)).count() as f64 / th.len() as f64);
                }
                if k % 8 == 0 {
                    n_search += 1;
                    for (d, &depth) in depths.iter().enumerate() {
                        let o = SearchOptions { width: 96, depth, ..Default::default() };
                        let t = Bot::new(w.clone(), o, 11).choose(&obs).unwrap().placement;
                        let t2 = Bot::new(w.clone(), o, 12).choose(&obs).unwrap().placement;
                        let nb = Bot::with_eval(NnueEval::new(net.clone()), o, 11).choose(&obs).unwrap().placement;
                        agree_net[d] += (nb == t) as usize;
                        agree_seed[d] += (t2 == t) as usize;
                        if depth == 6 && nb != t {
                            n_dis += 1;
                            for (col, pl) in [t, nb].into_iter().enumerate() {
                                let mut s = obs.me.hypothetical(&mut SplitMix(3));
                                let b2b = s.chain.b2b;
                                let o = s.apply_placement(pl);
                                let rows = [
                                    o.raw.total() > 0,
                                    o.lines > 0,
                                    o.broke_b2b && b2b >= 0,
                                    o.piece == tetris_sim::Piece::T && o.spin.is_spin() && o.lines > 0,
                                    o.tanked > 0,
                                ];
                                for (r, &on) in rows.iter().enumerate() {
                                    dis[r][col] += on as usize;
                                }
                            }
                        }
                    }
                }
            }
            let d = bots[p].choose(&obs).unwrap();
            m.play(p, d.placement);
        }
    }
    println!("{} positions from {games} games (depth-2 lines, hold included)", pooled.1);
    println!("Spearman(hand, net) inside one parent:  {:.3}", within.get());
    println!("Spearman(hand, net) pooled over lines:  {:.3}", pooled.get());
    println!("Spearman of the per-parent means:       {:.3}", parents.get());
    println!("teacher's top 32 lines in the net's top 32: {:.1}%", 100.0 * top_pooled.get());
    let wrong: usize = regret.iter().sum();
    println!(
        "one ply: net's best != teacher's best at {:.1}% of {n_one}; teacher regret mean {:.2} points;          < 1: {}, 1-5: {}, 5-20: {}, >= 20: {}",
        100.0 * wrong as f64 / n_one as f64,
        regret_sum / wrong.max(1) as f64,
        regret[0],
        regret[1],
        regret[2],
        regret[3]
    );
    println!("net minus teacher, centred within the position, in logits (1 point = {A} logits):");
    for (k, name) in ["no clear", "clear, no attack", "attack", "T-spin attack"].iter().enumerate() {
        println!("  {name:<17} n {:6}  mean {:+.3}  mean |.| {:.3}", resid[k].1, resid[k].get(), abs_resid[k].get());
    }
    println!("search agreement with the teacher at width 96 ({n_search} positions):");
    for (d, &depth) in depths.iter().enumerate() {
        println!(
            "  depth {depth}: net {:.1}%   teacher with another seed {:.1}%",
            100.0 * agree_net[d] as f64 / n_search as f64,
            100.0 * agree_seed[d] as f64 / n_search as f64
        );
    }
    println!("depth-6 disagreements ({n_dis}): share of moves that ...   teacher's move / net's move");
    for (r, name) in ["attack", "clear lines", "break B2B", "T-spin clear", "tank garbage"].iter().enumerate() {
        println!(
            "  {name:<12} {:5.1}% / {:5.1}%",
            100.0 * dis[r][0] as f64 / n_dis.max(1) as f64,
            100.0 * dis[r][1] as f64 / n_dis.max(1) as f64
        );
    }
}
