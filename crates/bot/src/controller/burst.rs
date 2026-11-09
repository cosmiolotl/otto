//! `my_burst(k)`: the most attack a player can generate in the next `k` placements, using only the
//! **known** queue (current, hold, preview). A narrow beam that maximises accumulated `raw` attack (surge
//! releases included); results are cached by the view's state hash, since the controller asks for both
//! players every move and a player's view often repeats (the opponent didn't move in between).

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use tetris_sim::movegen::{self, GenOptions};
use tetris_sim::{PREVIEW, Placement, PlayerState, SplitMix};

use crate::eval::board::board_features;
use crate::obs::PlayerView;
use crate::search::{PassThrough, state_hash};

#[derive(Debug, Default)]
pub struct Burst {
    beam: Vec<(PlayerState, f32, f32)>,
    next: Vec<(PlayerState, f32, f32)>,
    buf: Vec<Placement>,
    cache: HashMap<u64, f32, BuildHasherDefault<PassThrough>>,
    hits: u64,
}

const CACHE_MAX: usize = 4096;

impl Burst {
    /// `my_burst(k)` of a view (k is capped at the known queue: current + preview = 6 placements, 5 if a
    /// first hold would consume a preview piece).
    pub fn of_view(&mut self, v: &PlayerView, k: usize, width: usize, fpp: u32) -> f32 {
        // The tail after the preview is never reached for k <= PREVIEW; a fixed seed keeps it
        // deterministic anyway.
        let s = v.hypothetical(&mut SplitMix(0x5eed));
        let mut key = state_hash(&s) ^ (k as u64) << 59 ^ (width as u64) << 52 ^ (fpp as u64) << 40;
        for (i, p) in v.preview.iter().enumerate() {
            key = key.rotate_left(7) ^ (*p as u64) << (3 * i);
        }
        key = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        if let Some(&b) = self.cache.get(&key) {
            self.hits += 1;
            return b;
        }
        let b = self.run(&s, k.min(PREVIEW), width, fpp);
        if self.cache.len() >= CACHE_MAX {
            self.cache.clear();
        }
        self.cache.insert(key, b);
        b
    }

    /// The sub-search on a state (no cache).
    pub fn run(&mut self, s: &PlayerState, k: usize, width: usize, fpp: u32) -> f32 {
        let gopts = GenOptions::new(movegen::Mode::Sonic, true);
        self.beam.clear();
        self.beam.push((s.clone(), 0.0, 0.0));
        let mut best = 0.0f32;
        for _ in 0..k {
            self.next.clear();
            for (node, attack, _) in &self.beam {
                self.buf.clear();
                movegen::generate(node, gopts, &mut self.buf);
                for &p in &self.buf {
                    let mut c = node.clone();
                    let out = c.apply_placement(p);
                    if out.topout {
                        continue;
                    }
                    c.tick_n(fpp);
                    let a = attack + out.raw.total() as f32;
                    best = best.max(a);
                    // Tie-break between equal attack: keep the stack usable.
                    let f = board_features(&c.board);
                    let key = a - 0.05 * f.holes as f32 - 0.01 * f.max_height as f32 + 0.3 * f.tslot;
                    self.next.push((c, a, key));
                }
            }
            if self.next.is_empty() {
                break;
            }
            let w = width.max(1).min(self.next.len());
            let cmp = |a: &(PlayerState, f32, f32), b: &(PlayerState, f32, f32)| b.2.total_cmp(&a.2);
            if w < self.next.len() {
                self.next.select_nth_unstable_by(w - 1, cmp);
                self.next.truncate(w);
            }
            self.next.sort_by(cmp);
            std::mem::swap(&mut self.beam, &mut self.next);
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::{Board, Piece};

    #[test]
    fn finds_a_tsd_and_a_surge() {
        let mut s = PlayerState::tl(1);
        s.board = Board::from_ascii(
            "
            ..#.......
            ##...#####
            ###.######
            ",
        );
        s.falling = s.spawn_falling(Piece::T).0;
        let mut b = Burst::default();
        assert!(b.run(&s, 1, 4, 24) >= 4.0, "TSD = 4");
        // With b2b 6 the TSD keeps the chain (4 + 1); breaking it with a plain clear releases 6.
        s.chain.b2b = 6;
        assert!(b.run(&s, 1, 4, 24) >= 5.0);
        let v = PlayerView::from_state(&s);
        let x = b.of_view(&v, 1, 4, 24);
        let y = b.of_view(&v, 1, 4, 24);
        assert_eq!(x, y);
        assert_eq!(b.hits, 1);
    }
}
