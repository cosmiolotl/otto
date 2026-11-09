//! Beam search over own placements.
//!
//! - Ply 0 expands the root's movegen output (`Mode::Sonic`, hold branch included); every later ply
//!   expands each beam node the same way. A child is `clone + apply_placement` (scored), and kept nodes
//!   are advanced by `tick_n(fpp)`, so garbage arrival, tanking on non-clearing locks, cancelling and the
//!   multiplier all happen inside the search.
//! - Children are first stored as light candidates (parent, placement, score, path terms). The best are
//!   materialised again in that order (apply + tick), deduplicated by a state hash (transpositions),
//!   until `width` are kept.
//! - Ties break deterministically: score, then root-move index, then generation order.
//! - Beyond the last ply, `continuations` sampled bag-consistent tails can extend each leaf greedily by
//!   `extra_plies` and average the result (K = 1 and 0 extra plies = score the preview leaf).
//! - All buffers live in [`Search`] and are reused across moves.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{PieceQueue, Placement, PlayerState};

use crate::eval::{Evaluator, PathAcc};

/// Search parameters (not tuned by M5; the width is the main cost knob).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchOptions {
    /// Beam width (nodes kept per ply).
    pub width: usize,
    /// Plies: current + preview (6).
    pub depth: usize,
    /// Movegen reachability model (D7: `Sonic`).
    pub gen_mode: movegen::Mode,
    /// Bag-consistent continuations sampled beyond the preview (K).
    pub continuations: usize,
    /// Greedy plies per continuation after the beam (0 = score the preview leaf).
    pub extra_plies: usize,
}

pub const DEFAULT_WIDTH: usize = 96;

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            width: DEFAULT_WIDTH,
            depth: 6,
            gen_mode: movegen::Mode::Sonic,
            continuations: 1,
            extra_plies: 0,
        }
    }
}

/// Hasher for keys that are already hashes.
#[derive(Default)]
pub struct PassThrough(u64);

impl Hasher for PassThrough {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8 | b as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

pub type HashSet64 = HashSet<u64, BuildHasherDefault<PassThrough>>;

#[inline(always)]
fn mix(h: u64, v: u64) -> u64 {
    let x = (h ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^ (x >> 29)
}

/// Transposition key: board rows, current, hold, chain, clutch flag, queue position, garbage queue.
pub fn state_hash(s: &PlayerState) -> u64 {
    let rows = s.board.rows();
    let top = s.board.top();
    let mut h = 0x243f_6a88_85a3_08d3u64 ^ top as u64;
    for c in rows[..top].chunks(4) {
        let w = c.iter().enumerate().fold(0u64, |w, (i, &r)| w | (r as u64) << (16 * i));
        h = mix(h, w);
    }
    let misc = s.current() as u64
        | (s.hold.map_or(7, |p| p as u64)) << 3
        | (s.hold_locked as u64) << 6
        | (s.last_was_clear as u64) << 7
        | ((s.chain.b2b + 1) as u64 & 0xff) << 8
        | ((s.chain.combo + 1) as u64 & 0xff) << 16
        | (s.queue.dealt() as u64) << 24
        | (s.topped_out as u64) << 56;
    h = mix(h, misc);
    for g in &s.garbage.items {
        h = mix(h, g.amount as u64 | (g.frame as u64) << 32);
    }
    mix(h, s.garbage.sent as u64)
}

#[derive(Clone, Debug)]
struct Node {
    state: PlayerState,
    acc: PathAcc,
    root: u16,
    score: f32,
}

#[derive(Clone, Copy, Debug)]
struct Cand {
    parent: u32,
    root: u16,
    p: Placement,
    /// Index in the parent's move list (deterministic tie-break).
    mi: u16,
    score: f32,
    acc: PathAcc,
}

/// Result of one search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchResult {
    pub placement: Placement,
    pub score: f32,
    /// Children generated (evaluated).
    pub nodes: u64,
}

/// Reusable search buffers.
#[derive(Debug, Default)]
pub struct Search {
    roots: Vec<Placement>,
    buf: Vec<Placement>,
    cur: Vec<Node>,
    next: Vec<Node>,
    cands: Vec<Cand>,
    seen: HashSet64,
}

/// Score for a child that topped out at ply `ply` (later deaths are less bad).
#[inline]
fn death_score(ply: usize) -> f32 {
    -1.0e6 + ply as f32 * 1000.0
}

impl Search {
    /// Run the beam search from `root`. `tails` are extra sampled queues for continuations 2..=K (each a
    /// full queue from the root, as from [`crate::PlayerView::hypothetical`]). Returns `None` if the root
    /// has no placement.
    pub fn run<E: Evaluator>(
        &mut self,
        root: &PlayerState,
        fpp: u32,
        eval: &E,
        opts: &SearchOptions,
        tails: &[PieceQueue],
    ) -> Option<SearchResult> {
        let gopts = GenOptions::new(opts.gen_mode, true);
        let discount = eval.discount();
        let width = opts.width.max(1);
        self.roots.clear();
        movegen::generate(root, gopts, &mut self.roots);
        if self.roots.is_empty() {
            return None;
        }
        let mut nodes = 0u64;
        // Best dead end: (score, root).
        let mut dead: Option<(f32, u16)> = None;
        let mut note_dead = |score: f32, r: u16| {
            if dead.is_none_or(|(s, dr)| score > s || (score == s && r < dr)) {
                dead = Some((score, r));
            }
        };

        self.cur.clear();
        self.cur.push(Node { state: root.clone(), acc: PathAcc::default(), root: u16::MAX, score: 0.0 });
        let mut scratch = root.clone();
        for ply in 0..opts.depth.max(1) {
            self.cands.clear();
            for (pi, node) in self.cur.iter().enumerate() {
                let moves: &[Placement] = if ply == 0 {
                    &self.roots
                } else {
                    self.buf.clear();
                    movegen::generate(&node.state, gopts, &mut self.buf);
                    &self.buf
                };
                for (mi, &p) in moves.iter().enumerate() {
                    scratch.clone_from(&node.state);
                    let b2b0 = scratch.chain.b2b;
                    let out = scratch.apply_placement(p);
                    let mut acc = node.acc;
                    acc.push(&out, b2b0, discount);
                    let r = if ply == 0 { mi as u16 } else { node.root };
                    nodes += 1;
                    if out.topout {
                        note_dead(death_score(ply), r);
                        continue;
                    }
                    // The eval does not depend on the clock, so only kept nodes are ticked (below).
                    let score = eval.eval(&scratch, &acc);
                    self.cands.push(Cand { parent: pi as u32, root: r, p, mi: mi as u16, score, acc });
                }
            }
            if self.cands.is_empty() {
                break;
            }
            // Keep the best few × width (then dedupe and cut to width).
            let order = |a: &Cand, b: &Cand| {
                b.score
                    .total_cmp(&a.score)
                    .then(a.root.cmp(&b.root))
                    .then(a.parent.cmp(&b.parent))
                    .then(a.mi.cmp(&b.mi))
            };
            let keep = (4 * width).min(self.cands.len());
            if keep < self.cands.len() {
                self.cands.select_nth_unstable_by(keep - 1, order);
                self.cands.truncate(keep);
            }
            self.cands.sort_unstable_by(order);
            self.seen.clear();
            self.next.clear();
            for c in &self.cands {
                scratch.clone_from(&self.cur[c.parent as usize].state);
                scratch.apply_placement(c.p);
                scratch.tick_n(fpp);
                if !self.seen.insert(state_hash(&scratch)) {
                    continue; // transposition of a better node
                }
                self.next.push(Node { state: scratch.clone(), acc: c.acc, root: c.root, score: c.score });
                if self.next.len() == width {
                    break;
                }
            }
            std::mem::swap(&mut self.cur, &mut self.next);
        }

        if self.cur.first().is_none_or(|n| n.root == u16::MAX) {
            // Every line died at ply 0 (or no ply ran).
            let (score, r) = dead?;
            return Some(SearchResult { placement: self.roots[r as usize], score, nodes });
        }

        if opts.extra_plies > 0 {
            nodes += self.extend(fpp, eval, opts, root, tails);
        }

        let best = self
            .cur
            .iter()
            .min_by(|a, b| b.score.total_cmp(&a.score).then(a.root.cmp(&b.root)))
            .expect("non-empty beam");
        Some(SearchResult { placement: self.roots[best.root as usize], score: best.score, nodes })
    }

    /// Greedy extension of every leaf under K continuations; replaces each leaf's score by the mean.
    fn extend<E: Evaluator>(
        &mut self,
        fpp: u32,
        eval: &E,
        opts: &SearchOptions,
        root: &PlayerState,
        tails: &[PieceQueue],
    ) -> u64 {
        let gopts = GenOptions::new(opts.gen_mode, true);
        let discount = eval.discount();
        let k_total = 1 + tails.len();
        let mut nodes = 0;
        let base_dealt = root.queue.dealt();
        for li in 0..self.cur.len() {
            let mut sum = 0.0f32;
            for k in 0..k_total {
                let mut s = self.cur[li].state.clone();
                if k > 0 {
                    let mut q = tails[k - 1];
                    for _ in base_dealt..s.queue.dealt() {
                        q.pop();
                    }
                    s.queue = q;
                }
                let mut acc = self.cur[li].acc;
                let mut value = self.cur[li].score;
                for ply in 0..opts.extra_plies {
                    self.buf.clear();
                    movegen::generate(&s, gopts, &mut self.buf);
                    let mut best: Option<(f32, PlayerState, PathAcc)> = None;
                    for &p in &self.buf {
                        let mut c = s.clone();
                        let b2b0 = c.chain.b2b;
                        let out = c.apply_placement(p);
                        let mut a = acc;
                        a.push(&out, b2b0, discount);
                        nodes += 1;
                        let v = if out.topout {
                            death_score(opts.depth + ply)
                        } else {
                            c.tick_n(fpp);
                            eval.eval(&c, &a)
                        };
                        if best.as_ref().is_none_or(|b| v > b.0) {
                            best = Some((v, c, a));
                        }
                    }
                    let Some((v, c, a)) = best else {
                        value = death_score(opts.depth + ply);
                        break;
                    };
                    value = v;
                    if c.topped_out {
                        break;
                    }
                    s = c;
                    acc = a;
                }
                sum += value;
            }
            self.cur[li].score = sum / k_total as f32;
        }
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::HandEval;
    use crate::weights::Weights;

    #[test]
    fn deterministic_and_legal() {
        let eval = HandEval::new(Weights::default());
        let opts = SearchOptions { width: 16, ..Default::default() };
        let mut s = PlayerState::tl(77);
        let mut search = Search::default();
        let mut other = Search::default();
        for _ in 0..30 {
            let a = search.run(&s, 24, &eval, &opts, &[]).unwrap();
            let b = other.run(&s, 24, &eval, &opts, &[]).unwrap();
            assert_eq!(a, b);
            assert!(s.check_placement(&a.placement).is_ok());
            let out = s.apply_placement(a.placement);
            s.tick_n(24);
            assert!(!out.topout);
        }
        assert!(s.board.top() < 12, "a sane bot keeps the stack low on an empty field: {:?}", s.board);
    }

    #[test]
    fn hash_distinguishes_queue_position_and_garbage() {
        let s = PlayerState::tl(5);
        let mut t = s.clone();
        assert_eq!(state_hash(&s), state_hash(&t));
        t.receive_garbage(tetris_sim::IncomingGarbage { frame: 0, amount: 2, cid: 1, gameid: 2 });
        assert_ne!(state_hash(&s), state_hash(&t));
        let mut u = s.clone();
        u.hold = Some(tetris_sim::Piece::T);
        assert_ne!(state_hash(&s), state_hash(&u));
    }
}
