//! Feature sets `rows-v1` and `rows-v1-plan`: one player's visible state as a short list of active binary
//! features ([`Layout`]).
//!
//! All features are one-hot within their block, so a player has at most [`MAX_ACTIVE`] active features
//! (typically 40–60). Ordinal quantities (heights, B2B, pending garbage, …) are plain one-hot buckets here;
//! the trainer gives them thermometer-style "virtual" factor weights and folds them into these rows on
//! export. Since the first layer is linear, that is exactly equivalent to a thermometer encoding, at the
//! inference cost of one feature. The same trick factorises the row blocks: each `(row, pattern)`
//! weight is trained as `W[y, p] + W[p] + Σ_{x ∈ p} W[y, x] + W[y, popcount p]`.
//!
//! | Block | Size | Active | Content |
//! |---|---|---|---|
//! | `row` | 24 × 1024 | one per row below the stack top (≤ 24) | `(y, row bits)` |
//! | `covered` | 24 × 1024 | one per row with a covered hole | `(y, empty cells of row y with a filled cell above)` |
//! | `height` | 10 × 25 | 10 | `(x, min(height, 24))` |
//! | `step2` | 9 × 9 | 9 | `(x, clamp(h[x+1] - h[x], ±4))` |
//! | `step3` | 8 × 7 × 7 | 8 | `(x, clamp(h[x+1] - h[x], ±3), clamp(h[x+2] - h[x+1], ±3))` |
//! | `current` | 7 | 1 | falling piece |
//! | `hold` | 8 | 1 | hold piece or empty |
//! | `next` | 5 × 7 | 5 | preview slot × piece |
//! | `b2b` | 25 | 1 | `clamp(b2b + 1, 0, 24)` |
//! | `combo` | 13 | 1 | `clamp(combo + 1, 0, 12)` |
//! | `last_clear` | 1 | 0–1 | the last lock cleared lines |
//! | `pending` | 25 | 1 | incoming lines, `min(total, 24)` (plus lines the other player sent along a search path) |
//! | `ready` | 17 | 1 | incoming lines that may enter on the next lock, `min(·, 16)` |
//! | `clock` | 19 | 1 | `min(frame / 600, 18)` (10 s buckets: multiplier and cap growth) |
//! | `speed` | 29 | 1 | frames per piece, clamped to `12..=40` |
//!
//! **`rows-v1-plan`** is `rows-v1` (same indices) plus the side to move's release-controller [`Plan`] for the
//! current move, appended after [`N_FEATURES`] ([`push_plan`]). The opponent's half never has plan features.
//!
//! | Block | Size | Active | Content |
//! |---|---|---|---|
//! | `plan_mode` | 3 | 1 | BUILD, KILL, CANCEL |
//! | `plan_opp_burst` | 13 | 1 | the opponent's burst estimate in lines, `min(round(opp_burst), 12)` |
//! | `plan_opp_fill` | 6 | 1 | `u = 12 · opp_fill` (opponent height + pending − 8, clamped to 0..=12), bucket `ceil(5u / 12)` |
//!
//! Hidden information is never read: no queue entry past the preview, no RNG. The bag position is not
//! encoded, because search states' queues are not aligned with the real bag (`PieceQueue::from_prefix`).
//!
//! The same block table ([`BLOCKS`]) is written into every data file and network file, so the Python
//! trainer can check that it agrees with this layout.

use tetris_sim::board::{BOARD_WIDTH, FULL_ROW};
use tetris_sim::{PREVIEW, PlayerState};

use crate::controller::{Mode, Plan};

/// Feature-set name of `rows-v1` (stored in data and network files).
pub const FEATURE_SET: &str = "rows-v1";
/// Feature-set name of `rows-v1-plan`.
pub const FEATURE_SET_PLAN: &str = "rows-v1-plan";

pub const ROWS: usize = 24;
pub const PATTERNS: usize = 1 << BOARD_WIDTH;
pub const HEIGHT_MAX: usize = 24;
pub const B2B_MAX: usize = 24;
pub const COMBO_MAX: usize = 12;
pub const PENDING_MAX: usize = 24;
pub const READY_MAX: usize = 16;
pub const CLOCK_MAX: usize = 18;
pub const CLOCK_FRAMES: u32 = 600;
pub const FPP_MIN: u32 = 12;
pub const FPP_MAX: u32 = 40;

pub const ROW: usize = 0;
pub const COVERED: usize = ROW + ROWS * PATTERNS;
pub const HEIGHT: usize = COVERED + ROWS * PATTERNS;
pub const STEP2: usize = HEIGHT + BOARD_WIDTH * (HEIGHT_MAX + 1);
pub const STEP3: usize = STEP2 + 9 * 9;
pub const CURRENT: usize = STEP3 + 8 * 7 * 7;
pub const HOLD: usize = CURRENT + 7;
pub const NEXT: usize = HOLD + 8;
pub const B2B: usize = NEXT + PREVIEW * 7;
pub const COMBO: usize = B2B + B2B_MAX + 1;
pub const LAST_CLEAR: usize = COMBO + COMBO_MAX + 1;
pub const PENDING: usize = LAST_CLEAR + 1;
pub const READY: usize = PENDING + PENDING_MAX + 1;
pub const CLOCK: usize = READY + READY_MAX + 1;
pub const SPEED: usize = CLOCK + CLOCK_MAX + 1;
/// Total number of features per player.
pub const N_FEATURES: usize = SPEED + (FPP_MAX - FPP_MIN + 1) as usize;

/// Upper bound on active features per player: 24 + 24 rows, 10 + 9 + 8 surface, 1 + 1 + 5 pieces, 7 one-hots.
pub const MAX_ACTIVE: usize = ROWS + ROWS + BOARD_WIDTH + 9 + 8 + 1 + 1 + PREVIEW + 7;

// The plan block of `rows-v1-plan`, after every `rows-v1` feature.
pub const OPP_BURST_MAX: usize = 12;
pub const OPP_FILL_BUCKETS: usize = 6;
pub const PLAN_MODE: usize = N_FEATURES;
pub const PLAN_OPP_BURST: usize = PLAN_MODE + Mode::COUNT;
pub const PLAN_OPP_FILL: usize = PLAN_OPP_BURST + OPP_BURST_MAX + 1;
/// Total number of features per player in `rows-v1-plan`.
pub const N_FEATURES_PLAN: usize = PLAN_OPP_FILL + OPP_FILL_BUCKETS;
/// Active plan features: one per plan block.
pub const PLAN_ACTIVE: usize = 3;
/// Upper bound on active features per player in `rows-v1-plan` (and the capacity of [`Features`]).
pub const MAX_ACTIVE_PLAN: usize = MAX_ACTIVE + PLAN_ACTIVE;

/// One block of the layout: name, first index, shape (row-major).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub name: &'static str,
    pub offset: usize,
    pub dims: &'static [usize],
}

impl Block {
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub const BLOCKS: [Block; 15] = [
    Block { name: "row", offset: ROW, dims: &[ROWS, PATTERNS] },
    Block { name: "covered", offset: COVERED, dims: &[ROWS, PATTERNS] },
    Block { name: "height", offset: HEIGHT, dims: &[BOARD_WIDTH, HEIGHT_MAX + 1] },
    Block { name: "step2", offset: STEP2, dims: &[9, 9] },
    Block { name: "step3", offset: STEP3, dims: &[8, 7, 7] },
    Block { name: "current", offset: CURRENT, dims: &[7] },
    Block { name: "hold", offset: HOLD, dims: &[8] },
    Block { name: "next", offset: NEXT, dims: &[PREVIEW, 7] },
    Block { name: "b2b", offset: B2B, dims: &[B2B_MAX + 1] },
    Block { name: "combo", offset: COMBO, dims: &[COMBO_MAX + 1] },
    Block { name: "last_clear", offset: LAST_CLEAR, dims: &[1] },
    Block { name: "pending", offset: PENDING, dims: &[PENDING_MAX + 1] },
    Block { name: "ready", offset: READY, dims: &[READY_MAX + 1] },
    Block { name: "clock", offset: CLOCK, dims: &[CLOCK_MAX + 1] },
    Block { name: "speed", offset: SPEED, dims: &[(FPP_MAX - FPP_MIN + 1) as usize] },
];

/// The blocks `rows-v1-plan` appends to [`BLOCKS`].
pub const PLAN_BLOCKS: [Block; 3] = [
    Block { name: "plan_mode", offset: PLAN_MODE, dims: &[Mode::COUNT] },
    Block { name: "plan_opp_burst", offset: PLAN_OPP_BURST, dims: &[OPP_BURST_MAX + 1] },
    Block { name: "plan_opp_fill", offset: PLAN_OPP_FILL, dims: &[OPP_FILL_BUCKETS] },
];

/// A feature layout. A network file records its layout (as the hash of [`Layout::json`]), and a data set
/// records the layout its features were extracted with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Board, pieces, chain, garbage and clock of each player.
    #[default]
    RowsV1,
    /// `RowsV1` plus the side to move's controller plan ([`push_plan`]).
    RowsV1Plan,
}

impl Layout {
    pub const ALL: [Layout; 2] = [Layout::RowsV1, Layout::RowsV1Plan];

    pub const fn name(self) -> &'static str {
        match self {
            Layout::RowsV1 => FEATURE_SET,
            Layout::RowsV1Plan => FEATURE_SET_PLAN,
        }
    }

    /// Whether the side to move's features end with the plan block.
    pub const fn has_plan(self) -> bool {
        matches!(self, Layout::RowsV1Plan)
    }

    pub const fn n_features(self) -> usize {
        match self {
            Layout::RowsV1 => N_FEATURES,
            Layout::RowsV1Plan => N_FEATURES_PLAN,
        }
    }

    pub const fn max_active(self) -> usize {
        match self {
            Layout::RowsV1 => MAX_ACTIVE,
            Layout::RowsV1Plan => MAX_ACTIVE_PLAN,
        }
    }

    pub fn blocks(self) -> impl Iterator<Item = &'static Block> {
        let plan: &'static [Block] = if self.has_plan() { &PLAN_BLOCKS } else { &[] };
        BLOCKS.iter().chain(plan)
    }

    /// The block table as JSON (`{"feature_set": .., "n_features": .., "max_active": .., "blocks": [{name,
    /// offset, dims}]}`). The trainer reproduces this text byte for byte from the data's `STEM.json`.
    pub fn json(self) -> String {
        let blocks: Vec<String> = self
            .blocks()
            .map(|b| {
                let dims: Vec<String> = b.dims.iter().map(|d| d.to_string()).collect();
                format!(r#"{{"name":"{}","offset":{},"dims":[{}]}}"#, b.name, b.offset, dims.join(","))
            })
            .collect();
        format!(
            r#"{{"feature_set":"{}","n_features":{},"max_active":{},"blocks":[{}]}}"#,
            self.name(),
            self.n_features(),
            self.max_active(),
            blocks.join(",")
        )
    }

    /// FNV-1a of [`Layout::json`]: identifies the layout in network files.
    pub fn hash(self) -> u32 {
        let mut h = 0x811c_9dc5u32;
        for &b in self.json().as_bytes() {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
        h
    }

    /// The layout with this [`Layout::hash`].
    pub fn from_hash(hash: u32) -> Option<Layout> {
        Layout::ALL.into_iter().find(|l| l.hash() == hash)
    }
}

/// The `rows-v1` block table as JSON ([`Layout::json`]).
pub fn layout_json() -> String {
    Layout::RowsV1.json()
}

/// Active feature indices of one player (unsorted, no duplicates). Holds up to [`MAX_ACTIVE_PLAN`].
#[derive(Clone, Copy, Debug)]
pub struct Features {
    pub len: usize,
    pub idx: [u16; MAX_ACTIVE_PLAN],
}

impl Default for Features {
    fn default() -> Self {
        Features { len: 0, idx: [0; MAX_ACTIVE_PLAN] }
    }
}

impl Features {
    #[inline(always)]
    fn push(&mut self, i: usize) {
        debug_assert!(i < N_FEATURES_PLAN);
        self.idx[self.len] = i as u16;
        self.len += 1;
    }

    /// Append `extra` (for example [`plan_features`]).
    #[inline(always)]
    pub fn extend(&mut self, extra: &[u16]) {
        self.idx[self.len..self.len + extra.len()].copy_from_slice(extra);
        self.len += extra.len();
    }

    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        &self.idx[..self.len]
    }
}

const _: () = assert!(N_FEATURES_PLAN <= u16::MAX as usize, "feature indices are stored as u16");

/// The `rows-v1-plan` features of a plan (one per plan block; see the module docs for the buckets).
#[inline]
pub fn plan_features(plan: &Plan) -> [u16; PLAN_ACTIVE] {
    let burst = (plan.opp_burst.max(0.0).round() as usize).min(OPP_BURST_MAX);
    // opp_fill = (height + pending - 8) / 12, clamped: an integer u in 0..=12 over 12.
    let u = ((plan.opp_fill.clamp(0.0, 1.0) * 12.0).round() as usize).min(12);
    let fill = (u * (OPP_FILL_BUCKETS - 1)).div_ceil(12);
    [(PLAN_MODE + plan.mode.index()) as u16, (PLAN_OPP_BURST + burst) as u16, (PLAN_OPP_FILL + fill) as u16]
}

/// Append the plan features of `plan` to the side to move's features (layout `rows-v1-plan`).
#[inline]
pub fn push_plan(plan: &Plan, out: &mut Features) {
    out.extend(&plan_features(plan));
}

/// All `rows-v1-plan` features of the side to move `s` under `plan` ([`extract`] + [`push_plan`]).
pub fn extract_with_plan(s: &PlayerState, fpp: u32, plan: &Plan, out: &mut Features) {
    extract(s, fpp, out);
    push_plan(plan, out);
}

/// `pending` feature for `lines` incoming lines.
#[inline]
pub fn pending_feature(lines: u32) -> u16 {
    (PENDING + (lines as usize).min(PENDING_MAX)) as u16
}

/// Every feature of `s` except `pending` (see [`pending_feature`]). `fpp` is the player's frames per piece.
pub fn extract_static(s: &PlayerState, fpp: u32, out: &mut Features) {
    out.len = 0;
    let board = &s.board;
    let rows = board.rows();
    let top = board.top();

    let mut above = 0u16;
    for y in (0..top).rev() {
        let r = rows[y];
        if y < ROWS {
            out.push(ROW + y * PATTERNS + r as usize);
            let cov = above & !r & FULL_ROW;
            if cov != 0 {
                out.push(COVERED + y * PATTERNS + cov as usize);
            }
        }
        above |= r;
    }

    let h = board.heights();
    for x in 0..BOARD_WIDTH {
        out.push(HEIGHT + x * (HEIGHT_MAX + 1) + (h[x] as usize).min(HEIGHT_MAX));
    }
    let d = |x: usize, m: i32| (h[x + 1] as i32 - h[x] as i32).clamp(-m, m);
    for x in 0..BOARD_WIDTH - 1 {
        out.push(STEP2 + x * 9 + (d(x, 4) + 4) as usize);
    }
    for x in 0..BOARD_WIDTH - 2 {
        out.push(STEP3 + x * 49 + (d(x, 3) + 3) as usize * 7 + (d(x + 1, 3) + 3) as usize);
    }

    out.push(CURRENT + s.current().index());
    out.push(HOLD + s.hold.map_or(7, |p| p.index()));
    for i in 0..PREVIEW {
        out.push(NEXT + i * 7 + s.queue.peek(i).index());
    }

    out.push(B2B + ((s.chain.b2b + 1).max(0) as usize).min(B2B_MAX));
    out.push(COMBO + ((s.chain.combo + 1).max(0) as usize).min(COMBO_MAX));
    if s.last_was_clear {
        out.push(LAST_CLEAR);
    }

    // Garbage that can enter on the next lock (arrival + garbage speed ≤ the frame of that lock).
    let next_lock = s.frame.saturating_add(fpp);
    let ready: u32 = s.garbage.items.iter().filter(|g| g.ready_at(s.cfg) <= next_lock).map(|g| g.amount).sum();
    out.push(READY + (ready as usize).min(READY_MAX));

    out.push(CLOCK + ((s.frame / CLOCK_FRAMES) as usize).min(CLOCK_MAX));
    out.push(SPEED + (fpp.clamp(FPP_MIN, FPP_MAX) - FPP_MIN) as usize);
}

/// All features of `s`.
pub fn extract(s: &PlayerState, fpp: u32, out: &mut Features) {
    extract_static(s, fpp, out);
    out.push(pending_feature(s.garbage.size()) as usize);
}

/// Number of feature slots in [`Slots`] (the [`MAX_ACTIVE_PLAN`] slots, padded to a multiple of 16).
pub const SLOTS: usize = MAX_ACTIVE_PLAN.next_multiple_of(16);
/// An empty slot.
pub const SLOT_NONE: u16 = u16::MAX;
/// First slot of each block in [`Slots`].
pub const SLOT_ROW: usize = 0;
pub const SLOT_COVERED: usize = SLOT_ROW + ROWS;
pub const SLOT_HEIGHT: usize = SLOT_COVERED + ROWS;
pub const SLOT_STEP2: usize = SLOT_HEIGHT + BOARD_WIDTH;
pub const SLOT_STEP3: usize = SLOT_STEP2 + BOARD_WIDTH - 1;
pub const SLOT_CURRENT: usize = SLOT_STEP3 + BOARD_WIDTH - 2;
pub const SLOT_HOLD: usize = SLOT_CURRENT + 1;
pub const SLOT_NEXT: usize = SLOT_HOLD + 1;
pub const SLOT_B2B: usize = SLOT_NEXT + PREVIEW;
pub const SLOT_COMBO: usize = SLOT_B2B + 1;
pub const SLOT_LAST_CLEAR: usize = SLOT_COMBO + 1;
pub const SLOT_READY: usize = SLOT_LAST_CLEAR + 1;
pub const SLOT_CLOCK: usize = SLOT_READY + 1;
pub const SLOT_SPEED: usize = SLOT_CLOCK + 1;
pub const SLOT_PENDING: usize = SLOT_SPEED + 1;
pub const SLOT_PLAN: usize = SLOT_PENDING + 1;
const _: () = assert!(SLOT_PLAN == MAX_ACTIVE && SLOT_PLAN + PLAN_ACTIVE == MAX_ACTIVE_PLAN);

/// The same features as [`extract`] (+ the plan), each in a fixed slot: row `y` and covered row `y` have
/// slots of their own, then one slot per surface window, piece, and one-hot block, then the plan
/// ([`SLOT_PLAN`]). Inactive slots hold [`SLOT_NONE`]. Two states' feature sets then differ exactly in the
/// slots that differ, which is what an incremental accumulator update needs ([`Network::evaluate_cached`]).
///
/// [`Network::evaluate_cached`]: super::Network::evaluate_cached
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Slots(pub [u16; SLOTS]);

impl Default for Slots {
    fn default() -> Self {
        Slots([SLOT_NONE; SLOTS])
    }
}

impl Slots {
    /// The active features (in slot order).
    pub fn active(&self, out: &mut Features) {
        out.len = 0;
        for &f in &self.0 {
            if f != SLOT_NONE {
                out.push(f as usize);
            }
        }
    }
}

/// [`extract`] into slots, with `plan` (at most [`PLAN_ACTIVE`] features, for example [`plan_features`])
/// in the plan slots. Returns the number of active features.
#[inline]
pub fn extract_slots(s: &PlayerState, fpp: u32, plan: &[u16], out: &mut Slots) -> usize {
    debug_assert!(plan.len() <= PLAN_ACTIVE);
    let o = &mut out.0;
    *o = [SLOT_NONE; SLOTS];
    let board = &s.board;
    let rows = board.rows();
    let top = board.top();
    let mut n = 0;

    let mut above = 0u16;
    for y in (0..top).rev() {
        let r = rows[y];
        if y < ROWS {
            o[SLOT_ROW + y] = (ROW + y * PATTERNS + r as usize) as u16;
            let cov = above & !r & FULL_ROW;
            n += 1;
            if cov != 0 {
                o[SLOT_COVERED + y] = (COVERED + y * PATTERNS + cov as usize) as u16;
                n += 1;
            }
        }
        above |= r;
    }

    let h = board.heights();
    for x in 0..BOARD_WIDTH {
        o[SLOT_HEIGHT + x] = (HEIGHT + x * (HEIGHT_MAX + 1) + (h[x] as usize).min(HEIGHT_MAX)) as u16;
    }
    let d = |x: usize, m: i32| (h[x + 1] as i32 - h[x] as i32).clamp(-m, m);
    for x in 0..BOARD_WIDTH - 1 {
        o[SLOT_STEP2 + x] = (STEP2 + x * 9 + (d(x, 4) + 4) as usize) as u16;
    }
    for x in 0..BOARD_WIDTH - 2 {
        o[SLOT_STEP3 + x] = (STEP3 + x * 49 + (d(x, 3) + 3) as usize * 7 + (d(x + 1, 3) + 3) as usize) as u16;
    }
    n += BOARD_WIDTH + (BOARD_WIDTH - 1) + (BOARD_WIDTH - 2);

    o[SLOT_CURRENT] = (CURRENT + s.current().index()) as u16;
    o[SLOT_HOLD] = (HOLD + s.hold.map_or(7, |p| p.index())) as u16;
    for i in 0..PREVIEW {
        o[SLOT_NEXT + i] = (NEXT + i * 7 + s.queue.peek(i).index()) as u16;
    }
    o[SLOT_B2B] = (B2B + ((s.chain.b2b + 1).max(0) as usize).min(B2B_MAX)) as u16;
    o[SLOT_COMBO] = (COMBO + ((s.chain.combo + 1).max(0) as usize).min(COMBO_MAX)) as u16;
    n += 2 + PREVIEW + 2;
    if s.last_was_clear {
        o[SLOT_LAST_CLEAR] = LAST_CLEAR as u16;
        n += 1;
    }
    let next_lock = s.frame.saturating_add(fpp);
    let ready: u32 = s.garbage.items.iter().filter(|g| g.ready_at(s.cfg) <= next_lock).map(|g| g.amount).sum();
    o[SLOT_READY] = (READY + (ready as usize).min(READY_MAX)) as u16;
    o[SLOT_CLOCK] = (CLOCK + ((s.frame / CLOCK_FRAMES) as usize).min(CLOCK_MAX)) as u16;
    o[SLOT_SPEED] = (SPEED + (fpp.clamp(FPP_MIN, FPP_MAX) - FPP_MIN) as usize) as u16;
    o[SLOT_PENDING] = pending_feature(s.garbage.size());
    o[SLOT_PLAN..SLOT_PLAN + plan.len()].copy_from_slice(plan);
    n + 4 + plan.len()
}

/// Layer-stack bucket of the side to move: `min(stack height, 23) / 3` (8 buckets).
pub const N_BUCKETS: usize = 8;

#[inline]
pub fn bucket(s: &PlayerState) -> usize {
    s.board.top().min(23) / 3
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::{Board, IncomingGarbage};

    #[test]
    fn layout_is_contiguous() {
        let mut off = 0;
        for b in BLOCKS {
            assert_eq!(b.offset, off, "{}", b.name);
            off += b.len();
        }
        assert_eq!(off, N_FEATURES);
        for b in PLAN_BLOCKS {
            assert_eq!(b.offset, off, "{}", b.name);
            off += b.len();
        }
        assert_eq!(off, N_FEATURES_PLAN);
        assert!(layout_json().contains(r#""name":"speed""#));
        assert_eq!(Layout::RowsV1.blocks().count(), BLOCKS.len());
        assert_eq!(Layout::RowsV1Plan.blocks().count(), BLOCKS.len() + PLAN_BLOCKS.len());
        assert!(Layout::RowsV1Plan.json().starts_with(r#"{"feature_set":"rows-v1-plan","n_features":50076,"#));
        assert_ne!(Layout::RowsV1.hash(), Layout::RowsV1Plan.hash());
        for l in Layout::ALL {
            assert_eq!(Layout::from_hash(l.hash()), Some(l));
        }
    }

    /// The `rows-v1` text (and so its hash) is the one in every existing network and data set.
    #[test]
    fn rows_v1_layout_is_unchanged() {
        assert_eq!(Layout::RowsV1.json(), layout_json());
        assert!(layout_json().starts_with(r#"{"feature_set":"rows-v1","n_features":50054,"max_active":89,"#));
        assert!(!layout_json().contains("plan"));
    }

    #[test]
    fn plan_buckets() {
        let f = |mode, opp_burst, opp_fill| plan_features(&Plan { mode, opp_burst, opp_fill });
        assert_eq!(f(Mode::Build, 0.0, 0.0), [PLAN_MODE as u16, PLAN_OPP_BURST as u16, PLAN_OPP_FILL as u16]);
        let [m, b, l] = f(Mode::Cancel, 30.0, 1.0);
        assert_eq!((m as usize, b as usize, l as usize), (PLAN_MODE + 2, PLAN_OPP_BURST + 12, N_FEATURES_PLAN - 1));
        // Fill buckets of u = 12 · opp_fill: {0}, {1, 2}, {3, 4}, {5, 6, 7}, {8, 9}, {10, 11, 12}.
        let fill = |u: usize| f(Mode::Kill, 2.4, u as f32 / 12.0)[2] as usize - PLAN_OPP_FILL;
        assert_eq!((0..=12).map(fill).collect::<Vec<_>>(), [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5]);
        assert_eq!(f(Mode::Kill, 2.4, 0.0)[1] as usize - PLAN_OPP_BURST, 2);
        let plan = Plan { mode: Mode::Kill, opp_burst: 3.0, opp_fill: 0.5 };
        let (mut with, mut without) = (Features::default(), Features::default());
        extract_with_plan(&PlayerState::tl(1), 24, &plan, &mut with);
        extract(&PlayerState::tl(1), 24, &mut without);
        assert_eq!(&with.as_slice()[..without.len], without.as_slice());
        assert_eq!(&with.as_slice()[without.len..], plan_features(&plan));
    }

    #[test]
    fn rows_covered_and_surface() {
        let mut s = PlayerState::tl(1);
        s.board = Board::from_ascii(
            "
            ##........
            ..#.......
            ##.#######
            ",
        );
        let mut f = Features::default();
        extract(&s, 24, &mut f);
        let v = f.as_slice();
        // Row 0 = all but column 2; row 1 = column 2; row 2 = columns 0, 1.
        assert!(v.contains(&((ROW + 0b11_1111_1011) as u16)));
        assert!(v.contains(&((ROW + PATTERNS + 0b100) as u16)));
        assert!(v.contains(&((ROW + 2 * PATTERNS + 0b11) as u16)));
        // Row 1: columns 0 and 1 are empty under row 2's cells. Row 0: column 2 is under row 1's cell.
        assert!(v.contains(&((COVERED + PATTERNS + 0b11) as u16)));
        assert!(v.contains(&((COVERED + 0b100) as u16)));
        assert_eq!(v.iter().filter(|&&i| (i as usize) < HEIGHT).count(), 5);
        assert!(v.contains(&((HEIGHT + 2 * (HEIGHT_MAX + 1) + 2) as u16)));
        assert!(v.contains(&pending_feature(0)));
        let mut sorted = v.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), v.len(), "no duplicates");
    }

    #[test]
    fn garbage_and_hidden_info() {
        let mut s = PlayerState::tl(7);
        s.receive_garbage(IncomingGarbage { frame: 0, amount: 3, cid: 1, gameid: 2 });
        s.receive_garbage(IncomingGarbage { frame: 100, amount: 2, cid: 2, gameid: 2 });
        let mut f = Features::default();
        extract(&s, 24, &mut f);
        let v = f.as_slice();
        assert!(v.contains(&pending_feature(5)));
        // Item 1 arrives at 0 + 20 ≤ 24; item 2 at 120 > 24.
        assert!(v.contains(&((READY + 3) as u16)));
        // The queue tail and the garbage RNG do not matter.
        let mut t = s.clone();
        t.queue = s.queue.resample_tail(PREVIEW, 99);
        t.garbage.reseed(5);
        let mut g = Features::default();
        extract(&t, 24, &mut g);
        assert_eq!(f.as_slice(), g.as_slice());
    }

    /// [`extract_slots`] has exactly [`extract`]'s features (+ the plan), each in its block's slots.
    #[test]
    fn slots_hold_the_extracted_features() {
        let mut rng = tetris_sim::SplitMix(11);
        let plan = plan_features(&Plan { mode: Mode::Cancel, opp_burst: 4.0, opp_fill: 0.5 });
        for i in 0..400u32 {
            let mut s = PlayerState::tl(i as i64);
            let height = rng.below(34) as usize;
            let rows: Vec<u16> = (0..height).map(|_| rng.below(1024) as u16).collect();
            s.board = Board::from_rows(&rows, rng.below(1 << 20));
            s.chain.b2b = rng.below(40) as i32 - 1;
            s.chain.combo = rng.below(20) as i32 - 1;
            s.last_was_clear = i % 2 == 0;
            s.frame = rng.below(20_000) as u32;
            if i % 3 == 0 {
                s.hold = Some(tetris_sim::Piece::Z);
            }
            for k in 0..rng.below(4) {
                let frame = s.frame.saturating_sub(40) + 30 * k as u32;
                s.receive_garbage(IncomingGarbage { frame, amount: 1 + rng.below(9) as u32, cid: k as u32, gameid: 2 });
            }
            let fpp = 8 + rng.below(40) as u32;
            let extra: &[u16] = if i % 2 == 0 { &plan } else { &[] };
            let mut f = Features::default();
            extract(&s, fpp, &mut f);
            f.extend(extra);
            let mut slots = Slots::default();
            let n = extract_slots(&s, fpp, extra, &mut slots);
            let mut active = Features::default();
            slots.active(&mut active);
            assert_eq!(n, active.len);
            let (mut a, mut b) = (f.as_slice().to_vec(), active.as_slice().to_vec());
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "state {i}");
            // Each feature sits in its block's slot range.
            for (slot, &x) in slots.0.iter().enumerate() {
                let x = x as usize;
                let ok = match slot {
                    _ if x == SLOT_NONE as usize => true,
                    0..SLOT_COVERED => x == ROW + slot * PATTERNS + (x - ROW) % PATTERNS,
                    SLOT_COVERED..SLOT_HEIGHT => (COVERED..HEIGHT).contains(&x),
                    SLOT_HEIGHT..SLOT_STEP2 => (HEIGHT..STEP2).contains(&x),
                    SLOT_STEP2..SLOT_STEP3 => (STEP2..STEP3).contains(&x),
                    SLOT_STEP3..SLOT_CURRENT => (STEP3..CURRENT).contains(&x),
                    SLOT_PLAN.. => x >= N_FEATURES,
                    _ => (CURRENT..N_FEATURES).contains(&x),
                };
                assert!(ok, "slot {slot} holds {x}");
            }
        }
    }

    #[test]
    fn tall_messy_boards_stay_within_bounds() {
        let mut s = PlayerState::tl(3);
        let rows: Vec<u16> = (0..30).map(|y| if y % 2 == 0 { 0b10_1010_1010 } else { 0b01_0101_0101 }).collect();
        s.board = Board::from_rows(&rows, 0);
        s.chain.b2b = 40;
        s.chain.combo = 30;
        s.frame = 1_000_000;
        s.receive_garbage(IncomingGarbage { frame: 0, amount: 1000, cid: 1, gameid: 2 });
        let mut f = Features::default();
        extract(&s, 99, &mut f);
        assert!(f.len <= MAX_ACTIVE);
        assert!(f.as_slice().iter().all(|&i| (i as usize) < N_FEATURES));
        assert_eq!(bucket(&s), 7);
    }
}
