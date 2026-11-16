//! Beam search over own placements.
//!
//! - Ply 0 expands the root's movegen output ([`GEN_MODE`], hold branch included); every later ply
//!   expands each beam node the same way. A child is `clone + apply_placement`; a node's live children are
//!   scored together ([`Evaluator::eval_children`]), and kept nodes are advanced by `tick_n(fpp)`, so garbage arrival, tanking on non-clearing locks, cancelling and the
//!   multiplier all happen inside the search.
//! - Children are first stored as light candidates (parent, placement, score, path terms). The best are
//!   materialised again in that order (apply + tick), deduplicated by a state hash (transpositions),
//!   until `width` are kept.
//! - Ties break deterministically: score, then root-move index, then generation order.
//! - Beyond the last ply, `continuations` sampled bag-consistent tails can extend each leaf greedily by
//!   `extra_plies` and average the result (K = 1 and 0 extra plies = score the preview leaf).
//! - All buffers live in [`Search`] and are reused across moves.

use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{PieceQueue, Placement, PlayerState};

use crate::eval::{Evaluator, PathAcc};
use crate::hash::{HashSet64, state_hash};

/// The movegen reachability model of the search, the burst sub-search and the self-play legality checks
/// (`docs/DECISIONS.md` D7).
pub const GEN_MODE: movegen::Mode = movegen::Mode::Sonic;

/// Search parameters (not tuned by the M5 tuner; the width is the main cost knob).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchOptions {
    /// Beam width (nodes kept per ply).
    pub width: usize,
    /// Plies: current + preview (6).
    pub depth: usize,
    /// Bag-consistent continuations sampled beyond the preview (K).
    pub continuations: usize,
    /// Greedy plies per continuation after the beam (0 = score the preview leaf).
    pub extra_plies: usize,
}

pub const DEFAULT_WIDTH: usize = 96;

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions { width: DEFAULT_WIDTH, depth: 6, continuations: 1, extra_plies: 0 }
    }
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
    kids: Children,
}

/// One expanded node's live children, scored together: states (the first `len` are valid; the buffer only
/// grows), paths, scores, and per child (placement, move index, root move).
#[derive(Debug, Default)]
struct Children {
    states: Vec<PlayerState>,
    paths: Vec<PathAcc>,
    scores: Vec<f32>,
    meta: Vec<(Placement, u16, u16)>,
    len: usize,
}

impl Children {
    fn clear(&mut self) {
        self.paths.clear();
        self.meta.clear();
        self.len = 0;
    }

    /// A slot holding a copy of `parent`, to become the next child.
    fn slot(&mut self, parent: &PlayerState) -> &mut PlayerState {
        if self.len == self.states.len() {
            self.states.push(parent.clone());
        } else {
            self.states[self.len].clone_from(parent);
        }
        &mut self.states[self.len]
    }

    /// Keep the child in the slot.
    fn push(&mut self, path: PathAcc, meta: (Placement, u16, u16)) {
        self.paths.push(path);
        self.meta.push(meta);
        self.len += 1;
    }

    fn score<E: Evaluator>(&mut self, eval: &E, parent: &PlayerState) {
        self.scores.resize(self.len, 0.0);
        eval.eval_children(parent, &self.states[..self.len], &self.paths, &mut self.scores);
    }
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
        self.run_impl(root, fpp, eval, opts, tails, None)
    }

    /// [`Search::run`] with the first placement forced to `first`: ply 0 expands only that root move, so
    /// the whole beam is spent on its subtree, and the result's score is the search value of playing it
    /// (the same quantity as `run`'s score had `run` chosen it). With `depth` 1 that is exactly
    /// `eval(child, path of the one lock)`. `None` if `first` is not in the root's movegen output
    /// ([`GEN_MODE`], hold branch included).
    pub fn run_forced<E: Evaluator>(
        &mut self,
        root: &PlayerState,
        fpp: u32,
        eval: &E,
        opts: &SearchOptions,
        tails: &[PieceQueue],
        first: Placement,
    ) -> Option<SearchResult> {
        self.run_impl(root, fpp, eval, opts, tails, Some(first))
    }

    fn run_impl<E: Evaluator>(
        &mut self,
        root: &PlayerState,
        fpp: u32,
        eval: &E,
        opts: &SearchOptions,
        tails: &[PieceQueue],
        first: Option<Placement>,
    ) -> Option<SearchResult> {
        let gopts = GenOptions::new(GEN_MODE, true);
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
                let kids = &mut self.kids;
                kids.clear();
                for (mi, &p) in moves.iter().enumerate() {
                    if ply == 0 && first.is_some_and(|f| f != p) {
                        continue;
                    }
                    let child = kids.slot(&node.state);
                    let b2b0 = child.chain.b2b;
                    let out = child.apply_placement(p);
                    let mut acc = node.acc;
                    acc.push(&out, b2b0, discount);
                    let r = if ply == 0 { mi as u16 } else { node.root };
                    nodes += 1;
                    if out.topout {
                        note_dead(death_score(ply), r);
                        continue;
                    }
                    kids.push(acc, (p, mi as u16, r));
                }
                // The eval does not depend on the clock, so only kept nodes are ticked (below).
                kids.score(eval, &node.state);
                for k in 0..kids.len {
                    let ((p, mi, r), score, acc) = (kids.meta[k], kids.scores[k], kids.paths[k]);
                    self.cands.push(Cand { parent: pi as u32, root: r, p, mi, score, acc });
                }
            }
            if self.cands.is_empty() {
                break;
            }
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
        let gopts = GenOptions::new(GEN_MODE, true);
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
                    // Live children are scored together; the meta's move index marks them.
                    let kids = &mut self.kids;
                    kids.clear();
                    for (mi, &p) in self.buf.iter().enumerate() {
                        let c = kids.slot(&s);
                        let b2b0 = c.chain.b2b;
                        let out = c.apply_placement(p);
                        let mut a = acc;
                        a.push(&out, b2b0, discount);
                        nodes += 1;
                        if !out.topout {
                            c.tick_n(fpp);
                            kids.push(a, (p, mi as u16, 0));
                        }
                    }
                    kids.score(eval, &s);
                    // The first best in move order; a topped-out move scores `death_score` and ends the line.
                    let mut best: Option<(f32, Option<usize>)> = None;
                    let mut k = 0;
                    for mi in 0..self.buf.len() {
                        let live = k < kids.len && kids.meta[k].1 as usize == mi;
                        let v = if live { kids.scores[k] } else { death_score(opts.depth + ply) };
                        if best.is_none_or(|b| v > b.0) {
                            best = Some((v, live.then_some(k)));
                        }
                        k += live as usize;
                    }
                    let Some((v, live)) = best else {
                        value = death_score(opts.depth + ply);
                        break;
                    };
                    value = v;
                    let Some(k) = live else { break };
                    s.clone_from(&kids.states[k]);
                    acc = kids.paths[k];
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
    fn forced_first_move() {
        let eval = HandEval::new(Weights::default());
        let mut s = PlayerState::tl(31);
        let mut search = Search::default();
        let mut moves = Vec::new();
        for _ in 0..8 {
            moves.clear();
            movegen::generate(&s, GenOptions::new(GEN_MODE, true), &mut moves);
            // Depth 1: the forced value is the child's eval with its one-lock path, and the best of them
            // is the unforced search's.
            let one = SearchOptions { width: 16, depth: 1, ..Default::default() };
            let mut best = f32::NEG_INFINITY;
            for &p in &moves {
                let r = search.run_forced(&s, 24, &eval, &one, &[], p).unwrap();
                assert_eq!(r.placement, p);
                let mut c = s.clone();
                let b2b = c.chain.b2b;
                let out = c.apply_placement(p);
                if !out.topout {
                    let mut acc = PathAcc::default();
                    acc.push(&out, b2b, eval.discount());
                    assert_eq!(r.score, eval.eval(&c, &acc));
                }
                best = best.max(r.score);
            }
            assert_eq!(search.run(&s, 24, &eval, &one, &[]).unwrap().score, best);
            // Deeper: the forced move is the one returned, deterministically.
            let deep = SearchOptions { width: 8, depth: 4, ..Default::default() };
            let r = search.run_forced(&s, 24, &eval, &deep, &[], moves[moves.len() / 2]).unwrap();
            assert_eq!(r.placement, moves[moves.len() / 2]);
            assert_eq!(Search::default().run_forced(&s, 24, &eval, &deep, &[], r.placement), Some(r));
            let a = search.run(&s, 24, &eval, &deep, &[]).unwrap();
            s.apply_placement(a.placement);
            s.tick_n(24);
        }
        // A placement the root cannot make.
        let mut bad = moves[0];
        bad.y += 7;
        assert_eq!(search.run_forced(&s, 24, &eval, &SearchOptions::default(), &[], bad), None);
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
