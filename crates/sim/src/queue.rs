//! Piece queue fed by the 7-bag (`src/engine/queue/**`, RULES.md §11).
//!
//! A fixed ring buffer (`Copy`), refilled one bag at a time. The engine refills to 31 pieces; the
//! refill length does not change the sequence, only how far ahead it is generated.

use crate::piece::Piece;
use crate::rng::Rng;

const CAP: usize = 32;
/// Upcoming pieces a player sees (TL shows 5).
pub const PREVIEW: usize = 5;
/// The queue always holds at least this many upcoming pieces.
pub const MIN_PREVIEW: usize = 15;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PieceQueue {
    buf: [Piece; CAP],
    head: u8,
    len: u8,
    /// Pieces popped so far (bag-position bookkeeping; piece `i` of the sequence belongs to bag `i / 7`).
    dealt: u32,
    rng: Rng,
}

/// Number of already-popped pieces [`PieceQueue::recent`] can still return. The ring holds at most
/// `MIN_PREVIEW - 1 + 7` upcoming pieces, so the `CAP - 21` slots behind the head are never overwritten.
pub const RECENT: usize = CAP - (MIN_PREVIEW - 1 + 7);

/// Which pieces of one bag are already known (bag-position helper, see [`PieceQueue::bag_seen`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BagSeen {
    /// How many pieces of the bag come before the queried position (0..=6).
    pub offset: u8,
    /// Bit `Piece::index()` set ⇔ that piece of the bag comes before the queried position.
    pub mask: u8,
}

impl BagSeen {
    /// The pieces of the bag that are still to come, in `Piece::BAG_ORDER` order.
    pub fn unseen(&self) -> impl Iterator<Item = Piece> + '_ {
        Piece::BAG_ORDER.into_iter().filter(|p| self.mask >> p.index() & 1 == 0)
    }
}

impl PieceQueue {
    /// Queue for `seed` (bag RNG = `new RNG(seed)`).
    pub fn new(seed: i64) -> PieceQueue {
        let mut q = PieceQueue { buf: [Piece::I; CAP], head: 0, len: 0, dealt: 0, rng: Rng::new(seed) };
        q.refill();
        q
    }

    /// Queue whose upcoming pieces are exactly `prefix`, followed by 7-bags shuffled by `Rng::new(seed)`.
    ///
    /// Used for hypothetical (search) states: the known preview, then a sampled bag-consistent
    /// continuation. The caller is responsible for the prefix being bag-consistent (e.g. preview ++ a
    /// shuffle of [`BagSeen::unseen`]); the refill appends whole bags after it. `dealt()` starts at 0.
    /// Panics if `prefix.len() > CAP - 7`.
    pub fn from_prefix(prefix: &[Piece], seed: i64) -> PieceQueue {
        assert!(prefix.len() <= CAP - 7, "prefix too long");
        let mut q = PieceQueue { buf: [Piece::I; CAP], head: 0, len: 0, dealt: 0, rng: Rng::new(seed) };
        q.buf[..prefix.len()].copy_from_slice(prefix);
        q.len = prefix.len() as u8;
        q.refill();
        q
    }

    /// A copy that keeps the history (`dealt`, [`Self::recent`]) and the first `keep` upcoming pieces, and
    /// replaces everything after them by a random bag-consistent continuation from `Rng::new(seed)`: the
    /// rest of the bag that upcoming piece `keep` belongs to (a shuffle of the pieces of that bag not yet
    /// seen), then fresh 7-bags. Used to build states that differ only in hidden queue information.
    /// Needs `keep <= len()`.
    pub fn resample_tail(&self, keep: usize, seed: i64) -> PieceQueue {
        assert!(keep <= self.len());
        let seen = self.bag_seen(keep);
        let mut rest = [Piece::I; 7];
        let mut n = 0;
        for p in seen.unseen() {
            rest[n] = p;
            n += 1;
        }
        let mut rng = Rng::new(seed);
        rng.shuffle(&mut rest[..n]);
        let mut q = *self;
        q.len = keep as u8;
        for &p in &rest[..n] {
            let i = (q.head as usize + q.len as usize) % CAP;
            q.buf[i] = p;
            q.len += 1;
        }
        q.rng = rng;
        q.refill();
        q
    }

    fn push_bag(&mut self) {
        let mut bag = Piece::BAG_ORDER;
        self.rng.shuffle(&mut bag);
        for p in bag {
            let i = (self.head as usize + self.len as usize) % CAP;
            self.buf[i] = p;
            self.len += 1;
        }
    }

    #[inline]
    fn refill(&mut self) {
        while (self.len as usize) < MIN_PREVIEW {
            self.push_bag();
        }
    }

    /// Take the next piece.
    #[inline]
    pub fn pop(&mut self) -> Piece {
        self.refill();
        let p = self.buf[self.head as usize];
        self.head = ((self.head as usize + 1) % CAP) as u8;
        self.len -= 1;
        self.dealt += 1;
        self.refill();
        p
    }

    /// Number of pieces popped so far (for a player: the pieces locked + the current piece + the hold
    /// piece, if any).
    #[inline]
    pub fn dealt(&self) -> u32 {
        self.dealt
    }

    /// The `k`-th most recently popped piece (`k = 1` is the last one), or `None` if fewer than `k` were
    /// popped or `k > RECENT`.
    #[inline]
    pub fn recent(&self, k: usize) -> Option<Piece> {
        if k == 0 || k > RECENT || k > self.dealt as usize {
            return None;
        }
        Some(self.buf[(self.head as usize + CAP - k) % CAP])
    }

    /// Bag-position helper: for the bag that contains upcoming piece `i` (0 = next), the pieces of that
    /// bag that come before it (already popped, or upcoming pieces `< i`). It uses only pieces that were
    /// dealt or are at upcoming indices `< i`, so `bag_seen(5)` is what a player who sees a 5-piece
    /// preview (and remembers what was dealt) knows. Bags start where `dealt() + i ≡ 0 (mod 7)`.
    pub fn bag_seen(&self, i: usize) -> BagSeen {
        let pos = self.dealt as usize + i;
        let offset = pos % 7;
        let mut mask = 0u8;
        for j in 1..=offset {
            // Sequence index pos - j: upcoming if >= dealt, else already popped.
            let p = if j <= i { self.peek(i - j) } else { self.recent(j - i).expect("bag history") };
            mask |= 1 << p.index();
        }
        BagSeen { offset: offset as u8, mask }
    }

    /// The `i`-th upcoming piece (0 = next). Panics if `i >= self.len()`.
    #[inline]
    pub fn peek(&self, i: usize) -> Piece {
        assert!(i < self.len as usize);
        self.buf[(self.head as usize + i) % CAP]
    }

    /// Number of generated upcoming pieces (≥ [`MIN_PREVIEW`]).
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterator over the first `n` upcoming pieces (`n <= len()`).
    pub fn iter(&self, n: usize) -> impl Iterator<Item = Piece> + '_ {
        (0..n.min(self.len())).map(move |i| self.peek(i))
    }

    /// The bag RNG state.
    pub fn rng_state(&self) -> u32 {
        self.rng.state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_1300715353_first_bag() {
        // RULES.md §11: seed 1300715353 → first bag I O J Z L S T (replay `full` event).
        let mut q = PieceQueue::new(1_300_715_353);
        let got: String = (0..7).map(|_| q.pop().to_char()).collect();
        assert_eq!(got, "iojzlst");
    }

    #[test]
    fn prefix_and_bag_helpers() {
        let mut q = PieceQueue::new(42);
        let seq: Vec<Piece> = (0..q.len()).map(|i| q.peek(i)).collect();
        for _ in 0..9 {
            q.pop();
        }
        assert_eq!(q.dealt(), 9);
        assert_eq!(q.recent(1), Some(seq[8]));
        assert_eq!(q.recent(9), Some(seq[0]));
        assert_eq!(q.recent(10), None);
        // Upcoming piece 5 is sequence index 14: the start of bag 2, nothing seen.
        assert_eq!(q.bag_seen(5), BagSeen { offset: 0, mask: 0 });
        // Upcoming piece 3 = index 12, bag 1 (indices 7..14): indices 7, 8 popped, 9..12 upcoming.
        let s = q.bag_seen(3);
        assert_eq!(s.offset, 5);
        let want = seq[7..12].iter().fold(0u8, |m, p| m | 1 << p.index());
        assert_eq!(s.mask, want);
        assert_eq!(s.unseen().count(), 2);
        // Resampling keeps history and the first `keep` pieces, and stays bag-consistent.
        for seed in 1..50 {
            let r = q.resample_tail(3, seed);
            assert_eq!(r.dealt(), 9);
            assert_eq!(r.recent(1), q.recent(1));
            for i in 0..3 {
                assert_eq!(r.peek(i), q.peek(i));
            }
            let mut r = r;
            let mut seen = [false; 7];
            for _ in 0..5 {
                seen[r.pop().index()] = true;
            }
            for &p in &seq[7..9] {
                seen[p.index()] = true;
            }
            assert!(seen.iter().all(|&s| s), "bag 1 is a permutation");
            for _ in 0..6 {
                let mut seen = [false; 7];
                for _ in 0..7 {
                    seen[r.pop().index()] = true;
                }
                assert!(seen.iter().all(|&s| s));
            }
        }
        let p = PieceQueue::from_prefix(&[Piece::T, Piece::T, Piece::S], 5);
        assert_eq!((p.peek(0), p.peek(1), p.peek(2)), (Piece::T, Piece::T, Piece::S));
        assert!(p.len() >= MIN_PREVIEW);
        let mut seen = [false; 7];
        for i in 3..10 {
            seen[p.peek(i).index()] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn bags_are_permutations() {
        let mut q = PieceQueue::new(42);
        for _ in 0..50 {
            let mut seen = [false; 7];
            for _ in 0..7 {
                seen[q.pop().index()] = true;
            }
            assert!(seen.iter().all(|&s| s));
        }
    }
}
