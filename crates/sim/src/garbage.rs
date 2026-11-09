//! Incoming garbage queue: receive, FIFO cancel with opener phase, tank with cap, hole-column RNG
//! (RULES.md §9, §10; `src/engine/garbage/index.ts`, current `GarbageQueue`).

use smallvec::SmallVec;

use crate::board::BOARD_WIDTH;
use crate::config::Config;
use crate::rng::Rng;

/// `IncomingGarbage` (size is always 1, confirmed always true in the offline model).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IncomingGarbage {
    /// Frame the item started travelling (receiver's clock).
    pub frame: u32,
    pub amount: u32,
    pub cid: u32,
    pub gameid: u32,
}

/// Hole size (`garbageholesize`); only 1 is supported.
const HOLE_SIZE: usize = 1;
const COLUMN_WIDTH: u32 = (BOARD_WIDTH - (HOLE_SIZE - 1)) as u32;

#[derive(Clone, PartialEq, Debug)]
pub struct GarbageQueue {
    pub items: SmallVec<[IncomingGarbage; 6]>,
    pub last_column: Option<u8>,
    pub has_changed_column: bool,
    /// Opener-phase bookkeeping (`GarbageQueue.sent`): the remainder returned by each `cancel()` call.
    pub sent: u32,
    pub rng: Rng,
}

impl GarbageQueue {
    /// Queue with the garbage RNG `new RNG(seed)`.
    pub fn new(seed: i64) -> GarbageQueue {
        GarbageQueue {
            items: SmallVec::new(),
            last_column: None,
            has_changed_column: false,
            sent: 0,
            rng: Rng::new(seed),
        }
    }

    /// Replace the hole-column RNG by `Rng::new(seed)` (hypothetical search states: future hole columns
    /// are unknown to a player). A pre-rolled `last_column` is part of that hidden future, so it is
    /// re-rolled from the new RNG too (keeping "a column exists"). The items, `sent` and
    /// `has_changed_column` are unchanged.
    pub fn reseed(&mut self, seed: i64) {
        self.rng = Rng::new(seed);
        if self.last_column.is_some() {
            self.last_column = Some(self.rng.next_below(COLUMN_WIDTH) as u8);
        }
    }

    /// Total pending lines (`size`).
    #[inline]
    pub fn size(&self) -> u32 {
        self.items.iter().map(|g| g.amount).sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `receive`: append; items with `amount <= 0` are ignored. (The absolute cap is "none" in TL.)
    #[inline]
    pub fn receive(&mut self, item: IncomingGarbage) {
        if item.amount > 0 {
            self.items.push(item);
        }
    }

    /// `#reroll_column`.
    #[inline]
    fn reroll(&mut self, cfg: &Config) -> u8 {
        let cb: u32 = if cfg.messiness_center { (BOARD_WIDTH as f64 / 5.0).round() as u32 } else { 0 };
        let col = match self.last_column {
            Some(last) if cfg.messiness_nosame => {
                let mut c = cb + self.rng.next_below(COLUMN_WIDTH - 1 - 2 * cb);
                if c >= last as u32 {
                    c += 1;
                }
                c
            }
            _ => cb + self.rng.next_below(COLUMN_WIDTH - 2 * cb),
        } as u8;
        self.last_column = Some(col);
        col
    }

    /// Front item used up: `queue.shift(); if (rng() < change) { reroll(); hasChangedColumn = true }`.
    #[inline]
    fn pop_front(&mut self, cfg: &Config) {
        self.items.remove(0);
        if self.rng.next_float() < cfg.messiness_change {
            self.reroll(cfg);
            self.has_changed_column = true;
        }
    }

    /// `cancel(amount, pieceCount)`: cancel FIFO from the front. During the opener phase
    /// (`pieceCount + 1 <= openerPhase` and queue size ≥ `sent`) every point cancels two lines.
    /// Returns `(left to send, incoming lines removed)`.
    pub fn cancel(&mut self, amount: u32, piece_count: u32, cfg: &Config) -> (u32, u32) {
        let mut send = amount;
        let mut extra = 0;
        // `pieceCount + 1 <= openerPhase`
        if piece_count < cfg.opener_phase && self.size() >= self.sent {
            extra = amount;
        }
        let mut removed = 0;
        while (send > 0 || extra > 0) && !self.items.is_empty() {
            self.items[0].amount -= 1;
            removed += 1;
            if self.items[0].amount == 0 {
                self.pop_front(cfg);
            }
            if send > 0 {
                send -= 1;
            } else {
                extra -= 1;
            }
        }
        self.sent += send;
        (send, removed)
    }

    /// `tank(frame, cap, hard)`: move up to `floor(min(cap, cap.max))` arrived lines into the board.
    /// `on_line(column, cid)` is called per line, oldest first. Returns the number of lines.
    pub fn tank(&mut self, frame: u32, cap: f64, hard: bool, cfg: &Config, mut on_line: impl FnMut(u8, u32)) -> u32 {
        if self.items.is_empty() {
            return 0;
        }
        // Stable sort by frame (items normally arrive in order already).
        if !self.items.windows(2).all(|w| w[0].frame <= w[1].frame) {
            self.items.sort_by_key(|g| g.frame);
        }
        // messiness.timeout is 0 in TL (never rerolls here).
        let lines = cap.min(cfg.garbage_cap_max).floor().max(0.0) as u32;
        let limit = if hard { frame as i64 } else { frame as i64 - 1 };
        let mut n = 0;
        while n < lines && !self.items.is_empty() {
            let item = self.items[0];
            if item.frame as i64 + cfg.garbage_speed as i64 > limit {
                break;
            }
            self.items[0].amount -= 1;
            // `(col === null || rng() < within) && !hasChangedColumn`: the within draw happens whenever
            // a column exists, even if hasChangedColumn short-circuits the reroll afterwards.
            let col = match self.last_column {
                None => {
                    if !self.has_changed_column {
                        self.reroll(cfg)
                    } else {
                        unreachable!("hasChangedColumn without a column")
                    }
                }
                Some(c) => {
                    if self.rng.next_float() < cfg.messiness_within && !self.has_changed_column {
                        self.reroll(cfg)
                    } else {
                        c
                    }
                }
            };
            on_line(col, item.cid);
            self.has_changed_column = false;
            n += 1;
            if self.items[0].amount == 0 {
                self.pop_front(cfg);
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_example_seed_777() {
        // RULES.md §9.3: seed 777, incoming [3, 7], one 8-line tank → [0,0,0,9,9,9,9,9].
        let cfg = Config::TL;
        let mut q = GarbageQueue::new(777);
        q.receive(IncomingGarbage { frame: 0, amount: 3, cid: 1, gameid: 2 });
        q.receive(IncomingGarbage { frame: 0, amount: 7, cid: 2, gameid: 2 });
        let mut cols = vec![];
        let n = q.tank(20, 8.0, true, &cfg, |c, _| cols.push(c));
        assert_eq!(n, 8);
        assert_eq!(cols, vec![0, 0, 0, 9, 9, 9, 9, 9]);
        assert_eq!(q.size(), 2);
        assert_eq!(q.items[0].cid, 2);
    }

    #[test]
    fn arrival_needs_speed_frames() {
        let cfg = Config::TL;
        let mut q = GarbageQueue::new(1);
        q.receive(IncomingGarbage { frame: 10, amount: 2, cid: 1, gameid: 2 });
        assert_eq!(q.tank(29, 8.0, true, &cfg, |_, _| {}), 0);
        assert_eq!(q.tank(30, 8.0, false, &cfg, |_, _| {}), 0); // gravity lock needs one more frame
        assert_eq!(q.tank(30, 8.0, true, &cfg, |_, _| {}), 2);
    }

    #[test]
    fn opener_phase_doubles_cancel() {
        // RULES §10: a TSD (4) against incoming 10 leaves 2 on piece 1, and 6 at pieces = 14.
        let cfg = Config::TL;
        let mut q = GarbageQueue::new(1);
        q.receive(IncomingGarbage { frame: 0, amount: 10, cid: 1, gameid: 2 });
        assert_eq!(q.cancel(4, 0, &cfg), (0, 8));
        assert_eq!(q.size(), 2);
        let mut q = GarbageQueue::new(1);
        q.receive(IncomingGarbage { frame: 0, amount: 10, cid: 1, gameid: 2 });
        assert_eq!(q.cancel(4, 14, &cfg), (0, 4));
        assert_eq!(q.size(), 6);
    }

    #[test]
    fn cancel_remainder_counts_as_sent() {
        let cfg = Config::TL;
        let mut q = GarbageQueue::new(1);
        q.receive(IncomingGarbage { frame: 0, amount: 2, cid: 1, gameid: 2 });
        assert_eq!(q.cancel(5, 20, &cfg), (3, 2));
        assert_eq!(q.sent, 3);
        assert!(q.is_empty());
    }
}
