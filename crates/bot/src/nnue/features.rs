//! Feature set `rows-v1`: one player's visible state as a short list of active binary features.
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
//! Hidden information is never read: no queue entry past the preview, no RNG. The bag position is not
//! encoded, because search states' queues are not aligned with the real bag (`PieceQueue::from_prefix`).
//!
//! The same block table ([`BLOCKS`]) is written into every data file and network file, so the Python
//! trainer can check that it agrees with this layout.

use tetris_sim::board::{BOARD_WIDTH, FULL_ROW};
use tetris_sim::{PREVIEW, PlayerState};

/// Feature-set name (stored in data and network files).
pub const FEATURE_SET: &str = "rows-v1";

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

/// The block table as JSON (`{"feature_set": .., "n_features": .., "blocks": [{name, offset, dims}]}`).
pub fn layout_json() -> String {
    let blocks: Vec<String> = BLOCKS
        .iter()
        .map(|b| {
            let dims: Vec<String> = b.dims.iter().map(|d| d.to_string()).collect();
            format!(r#"{{"name":"{}","offset":{},"dims":[{}]}}"#, b.name, b.offset, dims.join(","))
        })
        .collect();
    format!(
        r#"{{"feature_set":"{FEATURE_SET}","n_features":{N_FEATURES},"max_active":{MAX_ACTIVE},"blocks":[{}]}}"#,
        blocks.join(",")
    )
}

/// Active feature indices of one player (unsorted, no duplicates).
#[derive(Clone, Copy, Debug)]
pub struct Features {
    pub len: usize,
    pub idx: [u16; MAX_ACTIVE],
}

impl Default for Features {
    fn default() -> Self {
        Features { len: 0, idx: [0; MAX_ACTIVE] }
    }
}

impl Features {
    #[inline(always)]
    fn push(&mut self, i: usize) {
        debug_assert!(i < N_FEATURES);
        self.idx[self.len] = i as u16;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        &self.idx[..self.len]
    }
}

const _: () = assert!(N_FEATURES <= u16::MAX as usize, "feature indices are stored as u16");

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
        assert!(layout_json().contains(r#""name":"speed""#));
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
