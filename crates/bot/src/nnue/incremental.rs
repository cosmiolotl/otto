//! Incremental accumulators: a search node's accumulator from a similar node's.
//!
//! A beam search evaluates the children of one parent after the other, and they share most features with
//! the parent and with each other. So a child's accumulator can be a known accumulator plus the FT rows of
//! the features the child has and that one lacks, minus the rows of those it lacks. Two references are
//! kept ([`AccCache`]):
//!
//! - **The parent, as its children see it** ([`Network::prepare_children`]): the parent's features with
//!   the queue advanced by one piece and the chain of a lock that clears nothing. A child that does not hold
//!   and clears nothing then differs only around its placement (about 26 rows on network self-play,
//!   instead of about 50 for a refresh).
//! - **The last evaluated node**, usually the previous sibling: the better reference for children that
//!   hold from an empty hold slot (the queue advances by two pieces).
//!
//! Features are compared slot by slot ([`Slots`]: every feature has a fixed slot, so two nodes' feature sets
//! differ exactly in the slots that differ). The cheaper reference is used, and a node that differs from
//! both by at least its refresh (line clears shift the rows) is refreshed.
//!
//! The accumulator is i16 with wrapping adds, a group, so an updated accumulator is bit-identical to a
//! refresh, and so is every score.
//!
//! (Measured and dropped: prefetching the next child's new rows one child ahead. On the laptop it did not
//! pay for itself.)

use tetris_sim::{PREVIEW, PlayerState};

use super::OpponentTable;
use super::features::{self, COMBO, CURRENT, NEXT, PENDING_MAX, SLOT_NONE, SLOTS, Slots};
use super::features::{SLOT_COMBO, SLOT_CURRENT, SLOT_LAST_CLEAR, SLOT_NEXT};
use super::forward;
use super::kernels::Scalar;
use super::net::{Accumulator, L1_MAX, Network};

#[derive(Clone, Debug)]
struct Entry {
    slots: Slots,
    acc: Accumulator,
}

impl Default for Entry {
    fn default() -> Self {
        Entry { slots: Slots::default(), acc: Accumulator([0; L1_MAX]) }
    }
}

/// The last node and the one being evaluated.
const RING: usize = 2;

/// Reference accumulators for [`Network::evaluate_cached`] and [`Network::evaluate_children`]: the prepared
/// parent and the last evaluated node. A cache belongs to one network with unchanged weights:
/// [`AccCache::clear`] it before using it with another.
#[derive(Clone, Debug, Default)]
pub struct AccCache {
    /// The prepared parent ([`Network::prepare_children`]), if `has_base`.
    base: Box<Entry>,
    has_base: bool,
    /// The last nodes, `last` the most recent.
    ring: Box<[Entry; RING]>,
    last: Option<usize>,
    /// Statistics: accumulators updated from the parent, from the last node, and refreshed.
    pub from_parent: u64,
    pub from_last: u64,
    pub refreshed: u64,
}

impl AccCache {
    pub fn new() -> AccCache {
        AccCache::default()
    }

    /// Forget both references (the next evaluation refreshes).
    pub fn clear(&mut self) {
        self.has_base = false;
        self.last = None;
    }
}

/// Try the last node as the reference too when the parent's difference has at least this many rows (and
/// whenever the current piece differs from the parent's prediction: a hold from an empty hold slot).
const LAST_TRY: usize = 32;

/// The features in `new` that are not in `old` (`add`) and those in `old` that are not in `new` (`sub`):
/// the slots that differ. Returns the counts.
#[inline(always)]
fn diff(net: &Network, old: &Slots, new: &Slots, add: &mut [u16; SLOTS], sub: &mut [u16; SLOTS]) -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    if net.avx2 {
        // SAFETY: AVX2 and POPCNT support were detected at construction.
        return unsafe { super::kernels::avx2::diff(old, new, add, sub) };
    }
    let _ = net;
    diff_portable(old, new, add, sub)
}

/// Portable [`diff`]: four slots compared at a time; only groups that differ are looked at.
#[inline(always)]
pub(super) fn diff_portable(
    old: &Slots,
    new: &Slots,
    add: &mut [u16; SLOTS],
    sub: &mut [u16; SLOTS],
) -> (usize, usize) {
    let (mut na, mut ns) = (0, 0);
    for (o, n) in old.0.chunks_exact(4).zip(new.0.chunks_exact(4)) {
        let (o, n): (&[u16; 4], &[u16; 4]) = (o.try_into().unwrap(), n.try_into().unwrap());
        if o == n {
            continue;
        }
        for k in 0..4 {
            let (x, y) = (o[k], n[k]);
            let differ = x != y;
            sub[ns] = x;
            ns += (differ & (x != SLOT_NONE)) as usize;
            add[na] = y;
            na += (differ & (y != SLOT_NONE)) as usize;
        }
    }
    (na, ns)
}

/// The active features of `s`, in slot order. Returns the count.
#[inline(always)]
fn gather(s: &Slots, out: &mut [u16; SLOTS]) -> usize {
    let mut n = 0;
    for &f in &s.0 {
        out[n] = f;
        n += (f != SLOT_NONE) as usize;
    }
    n
}

impl Network {
    /// Prepare `cache` for the children of `parent` (the side to move, `my_fpp` frames per piece, `extra`
    /// as in [`Network::evaluate_with`]): the parent's features as a child that neither holds nor clears
    /// lines would see them, and their accumulator. Only a hint: any node can still be evaluated.
    pub fn prepare_children(&self, cache: &mut AccCache, parent: &PlayerState, my_fpp: u32, extra: &[u16]) {
        let e = &mut *cache.base;
        features::extract_slots(parent, my_fpp, extra, &mut e.slots);
        let s = &mut e.slots.0;
        s[SLOT_CURRENT] = (CURRENT + parent.queue.peek(0).index()) as u16;
        for i in 0..PREVIEW {
            s[SLOT_NEXT + i] = (NEXT + i * 7 + parent.queue.peek(i + 1).index()) as u16;
        }
        s[SLOT_COMBO] = COMBO as u16; // combo -1
        s[SLOT_LAST_CLEAR] = SLOT_NONE;
        let mut add = [0u16; SLOTS];
        let n = gather(&e.slots, &mut add);
        self.update_acc(&self.ft_b, &add[..n], &[], &mut e.acc);
        cache.has_base = true;
    }

    #[inline]
    fn update_acc(&self, src: &Accumulator, add: &[u16], sub: &[u16], dst: &mut Accumulator) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 support was detected at construction.
            return unsafe { super::kernels::avx2::update_acc(self, src, add, sub, dst) };
        }
        forward::update::<Scalar>(self, src, add, sub, dst)
    }

    /// [`Network::evaluate_with`], bit for bit, computing the side to move's accumulator from the cheaper of
    /// `cache`'s references (or by a refresh), and remembering this node as the last one.
    pub fn evaluate_cached(
        &self,
        cache: &mut AccCache,
        me: &PlayerState,
        my_fpp: u32,
        opp: &OpponentTable,
        sent: u32,
        extra: &[u16],
    ) -> f32 {
        debug_assert_eq!(extra.len(), if self.layout.has_plan() { features::PLAN_ACTIVE } else { 0 });
        let entry = cache.last.map_or(0, |l| 1 - l);
        let [e0, e1] = &mut *cache.ring;
        let (last, dst) = if entry == 0 { (&*e1, e0) } else { (&*e0, e1) };
        let active = features::extract_slots(me, my_fpp, extra, &mut dst.slots);

        // The cheaper reference: (rows, adds, subs, accumulator).
        let (mut add, mut sub) = ([0u16; SLOTS], [0u16; SLOTS]);
        let mut best: (usize, usize, usize, Option<&Accumulator>) = (active, 0, 0, None);
        let mut try_last = cache.last.is_some();
        if cache.has_base {
            let (na, ns) = diff(self, &cache.base.slots, &dst.slots, &mut add, &mut sub);
            if na + ns < best.0 {
                best = (na + ns, na, ns, Some(&cache.base.acc));
            }
            try_last &= best.0 >= LAST_TRY || cache.base.slots.0[SLOT_CURRENT] != dst.slots.0[SLOT_CURRENT];
        }
        if try_last {
            let (mut add2, mut sub2) = ([0u16; SLOTS], [0u16; SLOTS]);
            let (na, ns) = diff(self, &last.slots, &dst.slots, &mut add2, &mut sub2);
            if na + ns < best.0 {
                best = (na + ns, na, ns, Some(&last.acc));
                (add, sub) = (add2, sub2);
            }
        }
        let (_, na, ns, src) = best;
        let (src, na) = match src {
            Some(src) => {
                if std::ptr::eq(src, &cache.base.acc) {
                    cache.from_parent += 1;
                } else {
                    cache.from_last += 1;
                }
                (src, na)
            }
            None => {
                cache.refreshed += 1;
                (&*self.ft_b, gather(&dst.slots, &mut add))
            }
        };
        let (add, sub) = (&add[..na], &sub[..ns]);
        let pending = (opp.base_pending + sent).min(PENDING_MAX as u32) as usize;
        let bucket = features::bucket(me);
        let acc = &mut dst.acc;
        cache.last = Some(entry);
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 support was detected at construction.
            let raw = unsafe { super::kernels::avx2::eval_update(self, src, add, sub, acc, opp, pending, bucket) };
            return Network::logit(raw);
        }
        Network::logit(forward::eval_update::<Scalar>(self, src, add, sub, acc, opp, pending, bucket))
    }

    /// The children of `parent` (each the parent after one lock), evaluated in order into `out`:
    /// [`Network::prepare_children`], then [`Network::evaluate_cached`] for each (`sent(i)` is child `i`'s
    /// `sent`). Bit-identical to [`Network::evaluate_with`].
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_children(
        &self,
        cache: &mut AccCache,
        parent: &PlayerState,
        children: &[PlayerState],
        sent: impl Fn(usize) -> u32,
        my_fpp: u32,
        opp: &OpponentTable,
        extra: &[u16],
        out: &mut [f32],
    ) {
        assert_eq!(children.len(), out.len());
        if children.is_empty() {
            return;
        }
        self.prepare_children(cache, parent, my_fpp, extra);
        for (i, (c, o)) in children.iter().zip(out.iter_mut()).enumerate() {
            *o = self.evaluate_cached(cache, c, my_fpp, opp, sent(i), extra);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{Mode, Plan};
    use crate::nnue::Layout;
    use crate::nnue::features::plan_features;
    use crate::nnue::net::SUPPORTED_L1;
    use tetris_sim::movegen::{self, GenOptions};
    use tetris_sim::{Board, IncomingGarbage, SplitMix};

    /// Parents with holes, garbage (some of it ready to enter), an occupied or empty hold slot, a combo, a
    /// B2B chain and an almost full board, so that their children clear lines (shifting the rows), take
    /// garbage, hold from an empty and from an occupied hold slot, and top out.
    fn parents() -> Vec<PlayerState> {
        let mut rng = SplitMix(5);
        let mut v = Vec::new();
        for i in 0..12u32 {
            let mut s = PlayerState::tl(i as i64 + 3);
            let height = [0, 2, 4, 7, 9, 12, 15, 18, 21, 3, 6, 30][i as usize];
            let rows: Vec<u16> = (0..height)
                .map(|y| if y % 3 == 0 { 0b11_1111_1111 & !(1 << rng.below(10)) } else { rng.below(1023) as u16 + 1 })
                .collect();
            s.board = Board::from_rows(&rows, if i % 2 == 0 { 0b11 } else { 0 });
            s.chain.b2b = i as i32 % 5 - 1;
            s.chain.combo = i as i32 % 4 - 1;
            s.last_was_clear = i % 3 == 0;
            s.frame = i * 900;
            if i % 2 == 1 {
                s.hold = Some(tetris_sim::Piece::T);
            }
            s.receive_garbage(IncomingGarbage { frame: 0, amount: i % 5, cid: 1, gameid: 2 });
            s.receive_garbage(IncomingGarbage { frame: s.frame + 300, amount: 2, cid: 2, gameid: 2 });
            v.push(s);
        }
        v
    }

    fn check(net: &Network) {
        let gopts = GenOptions::new(crate::GEN_MODE, true);
        let opp_state = PlayerState::tl(77);
        let opp = net.opponent_table(&opp_state, 26);
        let mut cache = AccCache::new();
        let (mut moves, mut children, mut sent) = (Vec::new(), Vec::new(), Vec::new());
        let (mut clears, mut holds, mut garbage) = (0, 0, 0);
        for (pi, parent) in parents().iter().enumerate() {
            let plan =
                Plan { mode: [Mode::Build, Mode::Kill, Mode::Cancel][pi % 3], opp_burst: pi as f32, opp_fill: 0.3 };
            let plan = plan_features(&plan);
            let extra: &[u16] = if net.layout.has_plan() { &plan } else { &[] };
            moves.clear();
            movegen::generate(parent, gopts, &mut moves);
            children.clear();
            sent.clear();
            for (mi, &p) in moves.iter().enumerate() {
                let mut c = parent.clone();
                let before = c.board.top();
                let out = c.apply_placement(p);
                if out.topout {
                    continue;
                }
                clears += (out.lines > 0) as usize;
                holds += p.hold as usize;
                garbage += (c.board.top() > before + 4) as usize;
                if mi % 3 == 0 {
                    c.tick_n(24);
                }
                children.push(c);
                sent.push(mi as u32 % 7);
            }
            let want: Vec<f32> =
                children.iter().zip(&sent).map(|(c, &k)| net.evaluate_with(c, 24, &opp, k, extra)).collect();
            // The batch, then one by one (from the batch's references), then from a stale parent.
            let mut got = vec![0.0; children.len()];
            net.evaluate_children(&mut cache, parent, &children, |i| sent[i], 24, &opp, extra, &mut got);
            assert_eq!(got, want, "evaluate_children, parent {pi}");
            for (i, c) in children.iter().enumerate().rev() {
                assert_eq!(net.evaluate_cached(&mut cache, c, 24, &opp, sent[i], extra), want[i], "parent {pi}");
            }
        }
        assert!(cache.from_parent > 0 && cache.from_last > 0 && cache.refreshed > 0, "{cache:?}");
        assert!(clears >= 8 && holds > 100 && garbage > 0, "coverage: {clears} clears, {holds} holds, {garbage}");
        // Another network: a cleared cache.
        cache.clear();
        let other = Network::random_layout(99, net.layout, net.l1);
        let c = &parents()[3];
        let extra = if net.layout.has_plan() { plan_features(&Plan::default()).to_vec() } else { Vec::new() };
        let t = other.opponent_table(&opp_state, 26);
        assert_eq!(other.evaluate_cached(&mut cache, c, 24, &t, 0, &extra), other.evaluate_with(c, 24, &t, 0, &extra));
    }

    #[test]
    fn incremental_matches_refresh() {
        for layout in Layout::ALL {
            for l1 in SUPPORTED_L1 {
                let net = Network::random_layout(l1 as u64 + 7, layout, l1);
                check(&net);
                let mut portable = net.clone();
                portable.use_portable_kernels();
                check(&portable);
            }
        }
    }

    #[test]
    fn diff_lists_the_changed_features() {
        let mut rng = SplitMix(3);
        for _ in 0..2000 {
            let (mut a, mut b) = (Slots::default(), Slots::default());
            for i in 0..SLOTS {
                let r = rng.below(4);
                a.0[i] = if rng.below(3) == 0 { SLOT_NONE } else { (i * 500 + rng.below(3) as usize) as u16 };
                b.0[i] = if r == 0 {
                    a.0[i]
                } else if r == 1 {
                    SLOT_NONE
                } else {
                    (i * 500 + rng.below(3) as usize) as u16
                };
            }
            let (mut add, mut sub) = ([0; SLOTS], [0; SLOTS]);
            let (na, ns) = diff_portable(&a, &b, &mut add, &mut sub);
            let want_add: Vec<u16> =
                (0..SLOTS).filter(|&i| a.0[i] != b.0[i] && b.0[i] != SLOT_NONE).map(|i| b.0[i]).collect();
            let want_sub: Vec<u16> =
                (0..SLOTS).filter(|&i| a.0[i] != b.0[i] && a.0[i] != SLOT_NONE).map(|i| a.0[i]).collect();
            assert_eq!((&add[..na], &sub[..ns]), (&want_add[..], &want_sub[..]));
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
                let (mut add2, mut sub2) = ([0; SLOTS], [0; SLOTS]);
                // SAFETY: AVX2 and POPCNT are available.
                let (na2, ns2) = unsafe { crate::nnue::kernels::avx2::diff(&a, &b, &mut add2, &mut sub2) };
                assert_eq!((&add2[..na2], &sub2[..ns2]), (&want_add[..], &want_sub[..]));
            }
        }
    }
}
