//! State hashing: the transposition key of the search and the burst cache.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

use tetris_sim::PlayerState;

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

pub(crate) type HashSet64 = HashSet<u64, BuildHasherDefault<PassThrough>>;

#[inline(always)]
fn mix(h: u64, v: u64) -> u64 {
    let x = (h ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^ (x >> 29)
}

/// Transposition key: board rows, current, hold and hold lock, chain, clutch flag (`last_was_clear`), queue
/// position, top-out, garbage queue and the opener `sent` counter.
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
