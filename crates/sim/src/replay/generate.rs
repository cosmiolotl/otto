//! Request generators for differential testing against the oracle (TRACE_FORMAT.md requests).
//!
//! - [`generate`]: placements from the reference BFS ([`crate::reach`]) with a shortest key path each,
//!   chosen by a cheap randomised heuristic. Styles bias play towards the rules that random play rarely
//!   reaches: spins and B2B chains (surge), all clears, garbage pressure, top-out/clutch, the multiplier.
//! - [`generate_movegen`]: placements from [`crate::movegen`] with [`Pathfinder`] key paths.

use super::request::{Request, RequestBuilder};
use crate::board::Board;
use crate::garbage::IncomingGarbage;
use crate::movegen::{self, GenOptions, Mode, Pathfinder};
use crate::reach::{Reachable, reachable};
use crate::rng::park_miller_seed;
use crate::state::{Key, Placement, PlayerState};
use crate::versus::Match;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// Two players, sane play, random unequal PPS.
    Versus,
    /// Two players chasing spins / B2B (surge releases).
    Spin,
    /// One player, all-clear search first, then sane play.
    AllClear,
    /// One player, stacks high under injected garbage (top-out, clutch).
    Topout,
    /// Two players, long initial wait so the multiplier is > 1.
    Long,
    /// One player, random noise keys mixed into the paths (exercises raw `press()` semantics).
    Noise,
    /// One player (no garbage) building B2B with quads in a right well, sometimes breaking it (surge).
    Quad,
}

impl Style {
    pub const ALL: [Style; 7] =
        [Style::Versus, Style::Spin, Style::AllClear, Style::Topout, Style::Long, Style::Noise, Style::Quad];

    pub fn name(self) -> &'static str {
        match self {
            Style::Versus => "versus",
            Style::Spin => "spin",
            Style::AllClear => "allclear",
            Style::Topout => "topout",
            Style::Long => "long",
            Style::Noise => "noise",
            Style::Quad => "quad",
        }
    }

    pub fn from_name(s: &str) -> Option<Style> {
        Style::ALL.into_iter().find(|x| x.name() == s)
    }
}

/// Small deterministic RNG (PCG-ish LCG) for generation choices.
#[derive(Clone, Debug)]
pub struct GenRng(u64);

impl GenRng {
    pub fn new(seed: u64) -> GenRng {
        GenRng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let x = self.0;
        (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 16
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() % 1_000_000) as f64 / 1e6 < p
    }
}

fn holes(b: &Board) -> i64 {
    let mut n = 0;
    for x in 0..10 {
        for y in 0..b.height(x) as usize {
            if !b.get(x, y) {
                n += 1;
            }
        }
    }
    n
}

/// El-Tetris-style board evaluation (×100), over columns `0..width` (columns ≥ `width` act as wall).
fn eltetris(b: &Board, width: usize, cells: &[(i8, i8); 4], lines: u8) -> i64 {
    let filled = |x: i32, y: i32| x < 0 || x >= width as i32 || y < 0 || b.occupied(x, y);
    let top = (0..width).map(|x| b.height(x) as i32).max().unwrap_or(0);
    let mut row_t = 0;
    for y in 0..top {
        for x in 0..=width as i32 {
            if filled(x - 1, y) != filled(x, y) {
                row_t += 1;
            }
        }
    }
    let mut col_t = 0;
    let mut holes = 0;
    let mut wells = 0;
    for x in 0..width as i32 {
        let h = b.height(x as usize) as i32;
        for y in 0..=h {
            if filled(x, y - 1) != filled(x, y) {
                col_t += 1;
            }
            if y < h && !filled(x, y) {
                holes += 1;
            }
        }
        let mut depth = 0;
        for y in (0..top + 1).rev() {
            if !filled(x, y) && filled(x - 1, y) && filled(x + 1, y) {
                depth += 1;
                wells += depth;
            } else {
                depth = 0;
            }
        }
    }
    let ys: Vec<i32> = cells.iter().map(|c| c.1 as i32).collect();
    let landing = (*ys.iter().min().unwrap() + *ys.iter().max().unwrap()) as f64 / 2.0;
    let v = -4.500 * landing + 3.418 * lines as f64
        - 3.218 * row_t as f64
        - 9.349 * col_t as f64
        - 7.899 * holes as f64
        - 3.386 * wells as f64;
    (v * 100.0) as i64
}

/// Score a candidate by applying it to a clone.
fn score(s: &PlayerState, r: &Reachable, style: Style, rng: &mut GenRng) -> i64 {
    let mut t = s.clone();
    let out = t.apply_placement(r.placement);
    let b = &t.board;
    let top = b.top() as i64;
    let width = if style == Style::Quad && top < 15 { 9 } else { 10 };
    let mut sc = eltetris(b, width, &out.cells, out.lines);
    sc += out.sent.total() as i64 * 300 + out.raw.total() as i64 * 100;
    if out.all_clear {
        sc += 1_000_000;
    }
    if out.topout {
        sc -= 10_000_000;
    }
    let mut bonus = 0i64;
    match style {
        Style::Spin => {
            if out.lines > 0 && out.spin.is_spin() {
                bonus += 400;
            }
            if out.lines > 0 && !out.spin.is_spin() && out.lines < 4 && s.chain.b2b >= 0 {
                bonus -= if s.chain.b2b >= 6 && rng.chance(0.3) { -300 } else { 250 };
            }
            if out.spin.is_spin() {
                bonus += 60;
            }
            bonus += t.chain.b2b.max(0) as i64 * 40;
        }
        Style::Topout => {
            sc = top * 20 - holes(b) * 2 + (rng.below(50) as i64);
            if out.lines > 0 {
                sc += 100;
            }
        }
        Style::Quad => {
            // Keep the well column empty until an I can take four lines; avoid breaking B2B unless the
            // chain is long (then sometimes break on purpose: surge).
            let well = 9; // right-side well
            let touches_well = out.cells.iter().any(|&(x, _)| x as usize == well);
            if out.lines == 4 {
                bonus += 3_000;
            } else if touches_well && top < 14 {
                bonus -= 800;
            }
            if out.lines > 0 && out.lines < 4 && !out.spin.is_spin() {
                bonus += if s.chain.b2b >= 4 && rng.chance(0.25) { 2_000 } else { -300 };
            }
            bonus += t.chain.b2b.max(0) as i64 * 50;
        }
        _ => {}
    }
    sc + bonus * 100 + rng.below(300) as i64
}

fn choose(s: &PlayerState, style: Style, rng: &mut GenRng) -> Reachable {
    let opts = reachable(s);
    if style != Style::Quad && rng.chance(0.05) {
        return opts[rng.below(opts.len() as u64) as usize].clone();
    }
    let mut best = 0;
    let mut best_sc = i64::MIN;
    for (i, r) in opts.iter().enumerate() {
        let sc = score(s, r, style, rng);
        if sc > best_sc {
            best_sc = sc;
            best = i;
        }
    }
    opts[best].clone()
}

/// Every 4-connected empty region below row `lim` must have a size divisible by 4.
fn regions_ok(b: &Board, lim: usize) -> bool {
    let mut seen = 0u64; // bit y*10+x, lim <= 6
    for y0 in 0..lim {
        for x0 in 0..10 {
            let bit = 1u64 << (y0 * 10 + x0);
            if b.get(x0, y0) || seen & bit != 0 {
                continue;
            }
            let mut stack = vec![(x0, y0)];
            seen |= bit;
            let mut n = 0;
            while let Some((x, y)) = stack.pop() {
                n += 1;
                let mut nb = |nx: usize, ny: usize| {
                    let bit = 1u64 << (ny * 10 + nx);
                    if !b.get(nx, ny) && seen & bit == 0 {
                        seen |= bit;
                        stack.push((nx, ny));
                    }
                };
                if x > 0 {
                    nb(x - 1, y);
                }
                if x < 9 {
                    nb(x + 1, y);
                }
                if y > 0 {
                    nb(x, y - 1);
                }
                if y + 1 < lim {
                    nb(x, y + 1);
                }
            }
            if n % 4 != 0 {
                return false;
            }
        }
    }
    true
}

/// Depth-first search for an all clear within `depth` pieces from the current state (hold allowed),
/// keeping the stack below `limit_rows` with no holes. Memoised on (board, pieces used, hold).
fn find_all_clear(
    s: &PlayerState,
    depth: u32,
    limit_rows: usize,
    budget: &mut u64,
    memo: &mut std::collections::HashSet<(u64, u32, Option<crate::Piece>)>,
) -> Option<Vec<Reachable>> {
    if depth == 0 || *budget == 0 {
        return None;
    }
    for r in reachable(s) {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let mut t = s.clone();
        let out = t.apply_placement(r.placement);
        if out.all_clear {
            return Some(vec![r]);
        }
        let b = &t.board;
        let lim = limit_rows.saturating_sub(out.lines as usize);
        if b.top() > lim || holes(b) > 0 || !regions_ok(b, lim) {
            continue;
        }
        let key = (
            b.rows()[..6].iter().enumerate().fold(0u64, |a, (y, &r)| a | (r as u64) << (y * 10)),
            t.stats.pieces + t.hold.is_some() as u32,
            t.hold,
        );
        if !memo.insert(key) {
            continue;
        }
        if let Some(mut rest) = find_all_clear(&t, depth - 1, lim, budget, memo) {
            rest.insert(0, r);
            return Some(rest);
        }
    }
    None
}

/// Insert random noise keys (never `Hold`/`HardDrop`) into a key path.
fn noisy(keys: &[Key], rng: &mut GenRng) -> Vec<Key> {
    const NOISE: [Key; 8] = [
        Key::MoveLeft,
        Key::MoveRight,
        Key::DasLeft,
        Key::DasRight,
        Key::SoftDrop,
        Key::RotateCw,
        Key::RotateCcw,
        Key::Rotate180,
    ];
    let mut out = Vec::new();
    for &k in keys {
        if k != Key::HardDrop && rng.chance(0.25) {
            out.push(NOISE[rng.below(8) as usize]);
        }
        out.push(k);
    }
    if rng.chance(0.3) {
        let n = out.len() - 1;
        out.insert(n, NOISE[rng.below(8) as usize]);
    }
    out
}

/// Generate one request. `pieces` is the per-player placement budget.
pub fn generate(style: Style, seed: i64, gen_seed: u64, pieces: u32) -> Request {
    let mut rng = GenRng::new(gen_seed);
    let players: usize = match style {
        Style::Versus | Style::Spin | Style::Long => 2,
        _ => 1,
    };
    let mut fpp = [0u32; 2];
    for f in &mut fpp {
        *f = 4 + rng.below(40) as u32;
    }
    if rng.chance(0.3) {
        fpp[1] = fpp[0];
    }
    let mut m = Match::tl(seed, fpp);
    let mut req = RequestBuilder::new(seed, players);
    let mut placed = [0u32; 2];

    if style == Style::Long {
        let t = 10_800 + rng.below(4_000) as u32;
        req.tick_to(t);
        m.advance_to(t);
        m.next_lock = [t + fpp[0], t + fpp[1]];
    }

    let mut queued_ac: Vec<Reachable> = Vec::new();
    if style == Style::AllClear {
        let mut budget = 400_000;
        let mut memo = std::collections::HashSet::new();
        let limit = if rng.chance(0.5) { 4 } else { 2 };
        if let Some(seq) = find_all_clear(&m.players[0], limit as u32 * 5 / 2, limit, &mut budget, &mut memo) {
            queued_ac = seq;
        }
    }

    loop {
        let p = if players == 1 {
            m.advance_to(m.next_lock[0]);
            0
        } else {
            match m.next_turn() {
                Some(p) => p,
                None => break,
            }
        };
        if placed[p] >= pieces || m.players[p].topped_out {
            break;
        }
        req.tick_to(m.frame);
        if players == 1 && style != Style::Quad && rng.chance(if style == Style::Topout { 0.35 } else { 0.15 }) {
            let n = 1 + rng.below(3);
            let amounts: Vec<u32> = (0..n).map(|_| 1 + rng.below(6) as u32).collect();
            for &a in &amounts {
                m.inject(p, a);
            }
            req.garbage(p, &amounts);
        }
        let r = if !queued_ac.is_empty() { queued_ac.remove(0) } else { choose(&m.players[p], style, &mut rng) };
        let keys = if style == Style::Noise { noisy(&r.keys, &mut rng) } else { r.keys.clone() };
        req.keys(p, &keys);
        let out = m.play_keys(p, &keys).expect("path ends in hardDrop");
        if players == 1 {
            m.next_lock[0] = m.frame + fpp[0];
        }
        placed[p] += 1;
        if out.topout {
            break;
        }
    }
    req.finish()
}

/// Placement counts of a [`generate_movegen`] request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MovegenCounts {
    pub placements: u32,
    pub spins: u32,
    pub holds: u32,
}

/// A one-player request whose key paths come from movegen (`Sonic`, with hold) and the pathfinder, for
/// checking the pathfinder against the oracle. Placements: low, few holes, often preferring spins.
pub fn generate_movegen(gen_seed: u64, pieces: u32) -> (Request, MovegenCounts) {
    let mut rng = GenRng::new(gen_seed);
    let match_seed = park_miller_seed(rng.next_u64());
    let mut s = PlayerState::tl(match_seed);
    let mut req = RequestBuilder::new(match_seed, 1);
    let mut pf = Pathfinder::new();
    let (mut buf, mut keys) = (Vec::new(), Vec::new());
    let mut n = MovegenCounts::default();
    for _ in 0..pieces {
        s.tick_n(20);
        req.tick_to(s.frame);
        if rng.below(8) == 0 {
            let amount = 1 + rng.below(4) as u32;
            req.garbage(0, &[amount]);
            s.receive_garbage(IncomingGarbage { frame: s.frame, amount, cid: 0, gameid: 0 });
        }
        buf.clear();
        movegen::generate(&s, GenOptions::new(Mode::Sonic, true), &mut buf);
        if buf.is_empty() {
            break;
        }
        let spin_bias = rng.below(2) == 0;
        let mut score = |p: &Placement| {
            let mut c = s.clone();
            let out = c.apply_placement(*p);
            let spin = if spin_bias && out.spin.is_spin() { 200 } else { 0 };
            c.board.top() as i64 * 40 + holes(&c.board) * 30 - out.lines as i64 * 20 - spin
                + out.topout as i64 * 100_000
                + rng.below(40) as i64
        };
        let p = *buf.iter().min_by_key(|p| score(p)).unwrap();
        keys.clear();
        assert!(pf.find(&s, &p, Mode::Sonic, &mut keys), "no path for {p:?}");
        let want = s.clone().apply_placement(p);
        assert_eq!(s.clone().apply_keys(&keys), Some(want));
        n.placements += 1;
        n.spins += want.spin.is_spin() as u32;
        n.holds += p.hold as u32;
        req.keys(0, &keys);
        if s.apply_placement(p).topout {
            break;
        }
    }
    (req.finish(), n)
}
