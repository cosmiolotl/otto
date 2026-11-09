//! What the bot is allowed to see, and the hypothetical search states built from it.
//!
//! [`PlayerState`] knows the whole future: the bag RNG (every future piece) and the garbage-hole RNG
//! (every future hole column). A real player sees neither. The bot only ever gets an [`Observation`]:
//!
//! - per player: board, falling piece, hold, `hold_locked`, the next [`PREVIEW`] pieces, which pieces of
//!   the bag after the preview were already seen ([`BagSeen`], from the preview and the dealt history),
//!   chain, `last_spin`, `last_was_clear`, the incoming garbage queue (amounts and arrival frames: the
//!   garbage meter), the opener-phase `sent` counter, `frame`, multiplier, cap, stats;
//! - not: the bag RNG, any piece after the preview, the garbage RNG, or the pre-rolled next hole column.
//!
//! [`PlayerView::hypothetical`] turns a view into a [`PlayerState`] for search: the known preview, then a
//! sampled bag-consistent continuation, and a garbage RNG re-seeded from the bot's own RNG.

use tetris_sim::config::Tracker;
use tetris_sim::garbage::GarbageQueue;
use tetris_sim::{
    BagSeen, Board, ChainState, Config, Falling, Match, PREVIEW, Piece, PieceQueue, PlayerState, Rng, Spin, SplitMix,
    Stats,
};

/// One player's visible state.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerView {
    pub cfg: &'static Config,
    pub board: Board,
    pub falling: Falling,
    pub hold: Option<Piece>,
    pub hold_locked: bool,
    pub preview: [Piece; PREVIEW],
    /// Pieces of the bag that the first piece after the preview belongs to, already seen.
    pub bag: BagSeen,
    pub chain: ChainState,
    pub last_spin: Spin,
    pub last_was_clear: bool,
    pub ihs_flag: bool,
    /// The incoming queue with its RNG blanked: `rng = Rng::new(1)`, and `last_column` (a pre-rolled future
    /// hole column) replaced by `Some(0)` / `None`.
    pub garbage: GarbageQueue,
    pub stats: Stats,
    pub frame: u32,
    pub multiplier: Tracker,
    pub cap: Tracker,
    pub topped_out: bool,
}

impl PlayerView {
    pub fn from_state(s: &PlayerState) -> PlayerView {
        let mut garbage = s.garbage.clone();
        garbage.rng = Rng::new(1);
        garbage.last_column = garbage.last_column.map(|_| 0);
        PlayerView {
            cfg: s.cfg,
            board: s.board,
            falling: s.falling,
            hold: s.hold,
            hold_locked: s.hold_locked,
            preview: std::array::from_fn(|i| s.queue.peek(i)),
            bag: s.queue.bag_seen(PREVIEW),
            chain: s.chain,
            last_spin: s.last_spin,
            last_was_clear: s.last_was_clear,
            ihs_flag: s.ihs_flag,
            garbage,
            stats: s.stats,
            frame: s.frame,
            multiplier: s.multiplier,
            cap: s.cap,
            topped_out: s.topped_out,
        }
    }

    #[inline]
    pub fn current(&self) -> Piece {
        self.falling.piece
    }

    /// Pending incoming lines.
    #[inline]
    pub fn pending(&self) -> u32 {
        self.garbage.size()
    }

    /// A search state consistent with this view: the preview, then the unseen rest of the current bag in a
    /// random order, then random 7-bags; the garbage RNG (and the pre-rolled column) re-seeded. All
    /// randomness comes from `rng`.
    pub fn hypothetical(&self, rng: &mut SplitMix) -> PlayerState {
        let mut prefix = [Piece::I; PREVIEW + 7];
        prefix[..PREVIEW].copy_from_slice(&self.preview);
        let mut n = PREVIEW;
        for p in self.bag.unseen() {
            prefix[n] = p;
            n += 1;
        }
        // Fisher–Yates on the unseen part.
        for i in (PREVIEW + 1..n).rev() {
            let j = PREVIEW + rng.below((i - PREVIEW + 1) as u64) as usize;
            prefix.swap(i, j);
        }
        let queue = PieceQueue::from_prefix(&prefix[..n], rng.park_miller_seed());
        let mut garbage = self.garbage.clone();
        garbage.reseed(rng.park_miller_seed());
        PlayerState {
            cfg: self.cfg,
            board: self.board,
            falling: self.falling,
            queue,
            hold: self.hold,
            hold_locked: self.hold_locked,
            chain: self.chain,
            last_spin: self.last_spin,
            last_was_clear: self.last_was_clear,
            garbage,
            stats: self.stats,
            frame: self.frame,
            multiplier: self.multiplier,
            cap: self.cap,
            topped_out: self.topped_out,
            ihs_flag: self.ihs_flag,
        }
    }
}

/// Everything the bot sees at its turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub me: PlayerView,
    pub opp: PlayerView,
    /// Own frames per piece (the bot's clock in the search).
    pub my_fpp: u32,
    /// The opponent's frames per piece (observable from their speed).
    pub opp_fpp: u32,
}

impl Observation {
    pub fn new(me: &PlayerState, opp: &PlayerState, my_fpp: u32, opp_fpp: u32) -> Observation {
        Observation { me: PlayerView::from_state(me), opp: PlayerView::from_state(opp), my_fpp, opp_fpp }
    }

    /// Player `p`'s observation of a two-player match.
    pub fn from_match(m: &Match, p: usize) -> Observation {
        Observation::new(&m.players[p], &m.players[1 - p], m.frames_per_piece[p], m.frames_per_piece[1 - p])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::IncomingGarbage;

    #[test]
    fn view_hides_rngs_and_hypothetical_is_consistent() {
        let mut s = PlayerState::tl(4242);
        for _ in 0..11 {
            let mut buf = Vec::new();
            tetris_sim::movegen::generate(&s, Default::default(), &mut buf);
            s.apply_placement(buf[buf.len() / 2]);
        }
        s.receive_garbage(IncomingGarbage { frame: s.frame, amount: 3, cid: 1, gameid: 2 });
        let mut t = s.clone();
        t.queue = s.queue.resample_tail(PREVIEW, 999);
        t.garbage.reseed(31337);
        assert_ne!(s, t);
        assert_eq!(PlayerView::from_state(&s), PlayerView::from_state(&t));

        let v = PlayerView::from_state(&s);
        let mut rng = SplitMix(7);
        for _ in 0..20 {
            let h = v.hypothetical(&mut rng);
            for i in 0..PREVIEW {
                assert_eq!(h.queue.peek(i), s.queue.peek(i));
            }
            // The bag after the preview is a permutation once the seen pieces are added.
            let mut mask = v.bag.mask;
            for i in PREVIEW..PREVIEW + (7 - v.bag.offset as usize) {
                let b = 1u8 << h.queue.peek(i).index();
                assert_eq!(mask & b, 0, "duplicate in bag");
                mask |= b;
            }
            assert_eq!(mask, 0x7f);
            assert_eq!(h.board, s.board);
            assert_eq!(h.garbage.items, s.garbage.items);
            // The real state's root moves are exactly the hypothetical's.
            let (mut a, mut b) = (Vec::new(), Vec::new());
            tetris_sim::movegen::generate(&s, tetris_sim::movegen::GenOptions::new(Default::default(), true), &mut a);
            tetris_sim::movegen::generate(&h, tetris_sim::movegen::GenOptions::new(Default::default(), true), &mut b);
            assert_eq!(a, b);
        }
    }
}
