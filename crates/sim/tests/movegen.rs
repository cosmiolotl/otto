//! Movegen correctness: fast movegen vs the brute-force `reach` BFS on random boards (both modes, with and
//! without hold), pathfinder paths (replay + shortest length + exact tie-break), `apply_placement ≡
//! apply_keys`, perft vs Fusion's reference counts, and fast perft vs brute-force perft.
//!
//! The default run is small enough for debug builds. The full corpus (10k boards × 7 pieces) is
//! `#[ignore]`d: `cargo test --release -p tetris-sim --test movegen -- --ignored`.
//! `MOVEGEN_BOARDS=N` overrides the board count of the default run.

use std::collections::{BTreeMap, BTreeSet};

use tetris_sim::board::{BOARD_HEIGHT, FULL_ROW};
use tetris_sim::movegen::{self, GenOptions, Mode, Pathfinder, PerftOpts, perft, perft_board, placement_cells};
use tetris_sim::reach::{ReachOpts, reachable_with};
use tetris_sim::{Board, Falling, Key, Piece, Placement, PlayerState, Rotation, Spin};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        (self.next() % 1_000_000) as f64 / 1_000_000.0 < p
    }
}

type Key3 = (bool, [(i8, i8); 4], Spin);

// ---------------------------------------------------------------------------------------------------
// Random boards
// ---------------------------------------------------------------------------------------------------

const STYLES: [&str; 7] = ["fill", "caves", "tslot", "garbage", "high", "noise", "towers"];

fn no_full_rows(rows: &mut [u16], rng: &mut Lcg) {
    for r in rows.iter_mut() {
        *r &= FULL_ROW;
        if *r == FULL_ROW {
            *r &= !(1 << rng.below(10));
        }
    }
}

/// Column heights with optional random holes below the surface.
fn heights_board(rng: &mut Lcg, max_h: usize, hole_p: f64) -> Vec<u16> {
    let mut rows = vec![0u16; BOARD_HEIGHT];
    let base = rng.below(max_h.max(1));
    for x in 0..10 {
        let h = (base as i32 + rng.below(7) as i32 - 3).clamp(0, max_h as i32) as usize;
        for row in rows.iter_mut().take(h) {
            if !rng.chance(hole_p) {
                *row |= 1 << x;
            }
        }
    }
    rows
}

fn random_board(rng: &mut Lcg, style: &str) -> Board {
    let mut rows = vec![0u16; BOARD_HEIGHT];
    match style {
        "fill" => rows = heights_board(rng, 12, 0.08),
        "caves" => {
            // Random overhang structure: fill, then carve horizontal tunnels and pockets.
            rows = heights_board(rng, 14, 0.0);
            for _ in 0..(2 + rng.below(5)) {
                let y = rng.below(12);
                let x0 = rng.below(10);
                let len = 1 + rng.below(6);
                for x in x0..(x0 + len).min(10) {
                    rows[y] &= !(1 << x);
                    if rng.chance(0.5) && y + 1 < BOARD_HEIGHT {
                        rows[y + 1] &= !(1 << x);
                    }
                }
            }
            // Roofs.
            for _ in 0..rng.below(4) {
                let y = 2 + rng.below(10);
                let x0 = rng.below(9);
                for x in x0..(x0 + 2 + rng.below(3)).min(10) {
                    rows[y] |= 1 << x;
                }
            }
        }
        "tslot" => {
            // A flat-ish stack with a T-spin slot (TSD / TST / mini shapes) cut into it.
            let h = 2 + rng.below(8);
            for row in rows.iter_mut().take(h) {
                *row = FULL_ROW;
            }
            let x = rng.below(8); // slot columns x..x+2
            let y = rng.below(h.max(1));
            match rng.below(4) {
                0 => {
                    // TSD: row y: hole at x+1; row y+1: holes x..x+2; overhang at x (or x+2) on row y+2.
                    rows[y] &= !(1 << (x + 1));
                    rows[y + 1] &= !(0b111 << x);
                    rows[y + 2] |= 1 << (if rng.chance(0.5) { x } else { x + 2 });
                    rows[y + 2] &= !(1 << (x + 1));
                }
                1 => {
                    // TST / fin slot: bar in column b (rows y..y+2), nub at (b-1, y+1), a blocker at
                    // (b, y+4) so only the (±1, 2) fin kick gets in; sometimes one bottom corner open
                    // (corner rule says mini, the fin kick forces normal); sometimes mirrored.
                    let b = 2 + rng.below(7);
                    for row in rows.iter_mut().take(y + 3) {
                        *row = FULL_ROW;
                    }
                    for k in 0..3 {
                        rows[y + k] &= !(1 << b);
                    }
                    rows[y + 1] &= !(1 << (b - 1));
                    if rng.chance(0.5) {
                        rows[y] &= !(1 << (b - 1));
                    }
                    for row in rows.iter_mut().skip(y + 3) {
                        *row = 0;
                    }
                    rows[y + 4] |= 1 << b;
                    if rng.chance(0.5) {
                        for row in rows.iter_mut() {
                            *row = row.reverse_bits() >> 6;
                        }
                    }
                    no_full_rows(&mut rows, rng);
                    return Board::from_rows(&rows, 0);
                }
                2 => {
                    // Mini/fin shapes: a 2-wide pocket under a roof.
                    rows[y] &= !(0b11 << x);
                    rows[y + 1] &= !(0b1 << x);
                    rows[y + 2] |= 0b11 << x;
                }
                _ => {
                    // Random notch.
                    for _ in 0..4 {
                        let yy = y + rng.below(3);
                        rows[yy] &= !(1 << (x + rng.below(3)));
                    }
                }
            }
            // Top it off with a bit of random surface.
            for x in 0..10 {
                if rng.chance(0.3) {
                    rows[h + 2] |= 1 << x;
                }
            }
        }
        "garbage" => {
            let n = 1 + rng.below(14);
            let mut hole = rng.below(10);
            for row in rows.iter_mut().take(n) {
                if rng.chance(0.3) {
                    hole = rng.below(10);
                }
                *row = FULL_ROW & !(1 << hole);
            }
            let top = heights_board(rng, 4, 0.1);
            for (i, r) in top.iter().enumerate() {
                if n + i < BOARD_HEIGHT {
                    rows[n + i] |= *r;
                }
            }
        }
        "high" => {
            // Near top-out: stacks up to (and sometimes into) the spawn rows.
            let max_h = 16 + rng.below(9);
            rows = heights_board(rng, max_h, 0.05);
            if rng.chance(0.3) {
                // Plug part of the spawn area.
                let y = 20 + rng.below(4);
                rows[y] |= 1 << (2 + rng.below(6));
            }
        }
        "noise" => {
            let h = 3 + rng.below(16);
            let p = 0.25 + rng.below(40) as f64 / 100.0;
            for row in rows.iter_mut().take(h) {
                for x in 0..10 {
                    if rng.chance(p) {
                        *row |= 1 << x;
                    }
                }
            }
        }
        _ => {
            // "towers": a few tall thin columns and wells.
            for x in 0..10 {
                if rng.chance(0.4) {
                    let h = rng.below(20);
                    for row in rows.iter_mut().take(h) {
                        *row |= 1 << x;
                    }
                }
            }
        }
    }
    no_full_rows(&mut rows, rng);
    Board::from_rows(&rows, 0)
}

/// A state with `board`, current piece `piece`, a random hold slot, random clutch flag and, sometimes, a
/// few random presses already made (mid-turn start with a nonzero `lastSpin`).
fn random_state(rng: &mut Lcg, board: Board, piece: Piece) -> PlayerState {
    let mut s = PlayerState::tl(1 + rng.below(1 << 20) as i64);
    s.board = board;
    s.last_was_clear = rng.chance(0.3);
    s.hold = if rng.chance(0.4) { None } else { Some(Piece::ALL[rng.below(7)]) };
    s.hold_locked = rng.chance(0.05);
    s.falling = s.spawn_falling(piece).0;
    s.last_spin = Spin::None;
    if rng.chance(0.15) {
        const K: [Key; 8] = [
            Key::MoveLeft,
            Key::MoveRight,
            Key::RotateCw,
            Key::RotateCcw,
            Key::Rotate180,
            Key::SoftDrop,
            Key::DasLeft,
            Key::DasRight,
        ];
        for _ in 0..1 + rng.below(4) {
            s.press(K[rng.below(K.len())]);
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------------------------------

fn movegen_set(s: &PlayerState, mode: Mode, hold: bool) -> (Vec<Placement>, BTreeSet<Key3>) {
    let mut v = Vec::new();
    movegen::generate(s, GenOptions::new(mode, hold), &mut v);
    let set: BTreeSet<Key3> = v.iter().map(|p| (p.hold, placement_cells(p), p.spin)).collect();
    assert_eq!(set.len(), v.len(), "movegen emitted duplicates\n{:?}\n{v:?}", s.board);
    (v, set)
}

fn reference(s: &PlayerState, mode: Mode, hold: bool, das: bool) -> BTreeMap<Key3, Vec<Key>> {
    let opts = ReachOpts { das, step: mode == Mode::Step, by_cells: true };
    reachable_with(s, opts, hold)
        .into_iter()
        .map(|r| ((r.placement.hold, placement_cells(&r.placement), r.placement.spin), r.keys))
        .collect()
}

fn describe(s: &PlayerState) -> String {
    format!(
        "{:?}\nfalling {:?} last_spin {:?} hold {:?} locked {} last_was_clear {}",
        s.board, s.falling, s.last_spin, s.hold, s.hold_locked, s.last_was_clear
    )
}

/// Movegen == reference set; every placement passes `check_placement`.
fn check_sets(s: &PlayerState, mode: Mode, hold: bool) -> usize {
    let (v, got) = movegen_set(s, mode, hold);
    let want: BTreeSet<Key3> = reference(s, mode, hold, false).into_keys().collect();
    if got != want {
        let extra: Vec<_> = got.difference(&want).collect();
        let missing: Vec<_> = want.difference(&got).collect();
        panic!("{mode:?} hold={hold} mismatch\n{}\nextra {extra:?}\nmissing {missing:?}", describe(s));
    }
    for p in &v {
        assert_eq!(s.check_placement(p), Ok(()), "{p:?}\n{}", describe(s));
    }
    v.len()
}

/// Pathfinder path for every placement: replays to the same outcome and state as `apply_placement`, and
/// equals the reference's shortest (lexicographically first) path.
fn check_paths(s: &PlayerState, mode: Mode, pf: &mut Pathfinder) {
    let (v, _) = movegen_set(s, mode, true);
    let shortest = reference(s, mode, true, true);
    let mut keys = Vec::new();
    for p in &v {
        keys.clear();
        assert!(pf.find(s, p, mode, &mut keys), "no path for {p:?}\n{}", describe(s));
        assert_eq!(keys.last(), Some(&Key::HardDrop));
        if mode == Mode::Sonic {
            assert!(!keys.contains(&Key::SoftDropStep));
        }
        let want = &shortest[&(p.hold, placement_cells(p), p.spin)];
        assert_eq!(&keys, want, "path for {p:?}\n{}", describe(s));
        let mut a = s.clone();
        let mut b = s.clone();
        let oa = a.apply_placement(*p);
        let ob = b.apply_keys(&keys).expect("path locks");
        assert_eq!(oa, ob, "outcome {p:?} keys {keys:?}\n{}", describe(s));
        assert_eq!(a, b, "state {p:?} keys {keys:?}");
    }
}

fn corpus(n_boards: usize, all_pieces: bool, seed: u64, mut f: impl FnMut(&PlayerState, bool)) {
    let mut rng = Lcg(seed);
    for i in 0..n_boards {
        let style = STYLES[i % STYLES.len()];
        let board = random_board(&mut rng, style);
        let pieces: Vec<Piece> = if all_pieces { Piece::ALL.to_vec() } else { vec![Piece::ALL[rng.below(7)]] };
        for (k, &p) in pieces.iter().enumerate() {
            let s = random_state(&mut rng, board, p);
            f(&s, (i + k) % 2 == 0);
        }
    }
}

fn env_boards(default: usize) -> usize {
    std::env::var("MOVEGEN_BOARDS").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------------

#[test]
fn movegen_matches_reference_small() {
    let n = env_boards(if cfg!(debug_assertions) { 140 } else { 1000 });
    let mut total = 0;
    corpus(n, false, 11, |s, hold| {
        for mode in [Mode::Sonic, Mode::Step] {
            total += check_sets(s, mode, hold);
        }
    });
    assert!(total > n);
}

#[test]
#[ignore = "10k boards × 7 pieces; run with --release -- --ignored"]
fn movegen_matches_reference_10k() {
    let mut states = 0usize;
    let mut placements = 0usize;
    let mut step_only = 0usize;
    corpus(10_000, true, 12345, |s, hold| {
        let a = check_sets(s, Mode::Sonic, hold);
        let b = check_sets(s, Mode::Step, hold);
        assert!(b >= a);
        if b > a {
            step_only += 1;
        }
        states += 1;
        placements += a + b;
    });
    eprintln!("checked {states} states, {placements} placements; Step ⊋ Sonic on {step_only} states");
}

#[test]
fn paths_small() {
    let n = env_boards(if cfg!(debug_assertions) { 30 } else { 300 });
    let mut pf = Pathfinder::new();
    corpus(n, false, 77, |s, _| {
        for mode in [Mode::Sonic, Mode::Step] {
            check_paths(s, mode, &mut pf);
        }
    });
}

#[test]
#[ignore = "slow; run with --release -- --ignored"]
fn paths_2k() {
    let mut pf = Pathfinder::new();
    corpus(2_000, true, 4242, |s, _| {
        for mode in [Mode::Sonic, Mode::Step] {
            check_paths(s, mode, &mut pf);
        }
    });
}

/// Movegen output fed to `apply_placement` in real games, both modes: always legal, and equal to the
/// pathfinder replay (`apply_keys`).
#[test]
fn placement_equals_keys_in_games() {
    let mut rng = Lcg(5);
    let mut pf = Pathfinder::new();
    let mut v = Vec::new();
    let mut keys = Vec::new();
    for game in 0..4 {
        let mut s = PlayerState::tl(300 + game);
        for i in 0..40 {
            if i % 5 == 0 {
                s.receive_garbage(tetris_sim::IncomingGarbage {
                    frame: s.frame,
                    amount: 1 + rng.below(3) as u32,
                    cid: i,
                    gameid: 2,
                });
            }
            s.tick_n(30);
            let mode = if i % 2 == 0 { Mode::Sonic } else { Mode::Step };
            v.clear();
            movegen::generate(&s, GenOptions::new(mode, true), &mut v);
            assert!(!v.is_empty());
            for p in &v {
                keys.clear();
                assert!(pf.find(&s, p, mode, &mut keys));
                let mut a = s.clone();
                let mut b = s.clone();
                assert_eq!(a.apply_placement(*p), b.apply_keys(&keys).unwrap());
                assert_eq!(a, b);
            }
            // Play a low placement.
            let p = *v.iter().min_by_key(|p| (p.y, rng.below(5))).unwrap();
            if s.apply_placement(p).topout {
                break;
            }
        }
    }
}

// --- Spin fixtures ---------------------------------------------------------------------------------

fn spins_at(board: &Board, piece: Piece, mode: Mode, cells: &[(i8, i8)]) -> BTreeSet<Spin> {
    let mut v = Vec::new();
    movegen::generate_from(board, Falling::spawn(piece), Spin::None, mode, false, &mut v);
    let mut want: Vec<(i8, i8)> = cells.to_vec();
    want.sort_unstable();
    v.iter().filter(|p| placement_cells(p).to_vec() == want).map(|p| p.spin).collect()
}

#[test]
fn tsd_slot_is_normal() {
    // Covered TSD slot: only reachable by rotating in, and both front corners are filled.
    let b = Board::from_ascii(
        "
        ##........
        #...######
        ##.#######
        ",
    );
    let s = spins_at(&b, Piece::T, Mode::Sonic, &[(1, 1), (2, 0), (2, 1), (3, 1)]);
    assert_eq!(s, BTreeSet::from([Spin::Normal]));
}

#[test]
fn t_on_flat_ground_has_none_and_maybe_spins() {
    let b = Board::new();
    let s = spins_at(&b, Piece::T, Mode::Sonic, &[(3, 0), (4, 0), (5, 0), (4, 1)]);
    assert_eq!(s, BTreeSet::from([Spin::None]));
}

#[test]
fn tst_fin_kick_forces_normal() {
    // TST slot: bar in column 2 (rows 0-2), nub at (1, 1). The T rests flat on row 3 with (2, 4) blocked
    // above test 1, so CCW from North fails tests 0-3 and the (1, 2) fin/TST kick succeeds. The corner
    // rule alone says mini (one front corner); the fin kick forces normal.
    let b = Board::from_ascii(
        "
        ..#.......
        ..........
        ##.#######
        #..#######
        #..#######
        ",
    );
    let start = Falling::new(Piece::T, Rotation::North, 0, 4);
    assert!(start.fits(&b) && start.is_grounded(&b));
    let mut f = start;
    let r = f.rotate(&b, -1).unwrap();
    assert_eq!((r.kick, r.spin), (Some((1, 2)), Spin::Normal));
    assert_eq!(tetris_sim::falling::t_corner_spin(&b, &f, false), Spin::Mini);
    let mut cells = f.cells();
    cells.sort_unstable();
    for mode in [Mode::Sonic, Mode::Step] {
        let mut v = Vec::new();
        movegen::generate_from(&b, start, Spin::None, mode, false, &mut v);
        let spins: BTreeSet<Spin> = v.iter().filter(|p| placement_cells(p) == cells).map(|p| p.spin).collect();
        assert!(spins.contains(&Spin::Normal), "{mode:?} {spins:?}");
    }
}

/// The random corpus exercises every spin path: T normal, T mini, fin-forced normal, non-T minis, and
/// "duals" (same cells reachable with different spin outcomes).
#[test]
fn corpus_covers_spin_cases() {
    let (mut t_normal, mut t_mini, mut fin_forced, mut other_mini, mut duals) = (0, 0, 0, 0, 0);
    corpus(env_boards(if cfg!(debug_assertions) { 150 } else { 2000 }), true, 99, |s, hold| {
        let (v, _) = movegen_set(s, Mode::Step, hold);
        let mut by_cells: BTreeMap<_, BTreeSet<Spin>> = BTreeMap::new();
        for p in &v {
            by_cells.entry((p.hold, placement_cells(p))).or_default().insert(p.spin);
            match (p.piece, p.spin) {
                (Piece::T, Spin::Normal) => {
                    t_normal += 1;
                    if tetris_sim::falling::t_corner_spin(&s.board, &p.falling(), false) != Spin::Normal {
                        fin_forced += 1;
                    }
                }
                (Piece::T, Spin::Mini) => t_mini += 1,
                (_, Spin::Mini) => other_mini += 1,
                _ => {}
            }
        }
        duals += by_cells.values().filter(|s| s.len() > 1).count();
    });
    eprintln!("T normal {t_normal}, T mini {t_mini}, fin-forced {fin_forced}, non-T mini {other_mini}, duals {duals}");
    assert!(t_normal > 0 && t_mini > 0 && fin_forced > 0 && other_mini > 0 && duals > 0);
}

#[test]
fn hold_branch_flags_and_piece() {
    let mut s = PlayerState::tl(1_300_715_353); // I O J Z L S T ...
    assert_eq!(s.current(), Piece::I);
    let mut v = Vec::new();
    movegen::generate(&s, GenOptions::new(Mode::Sonic, true), &mut v);
    assert!(v.iter().any(|p| p.hold && p.piece == Piece::O));
    assert!(v.iter().any(|p| !p.hold && p.piece == Piece::I));
    // On the empty board: I 17 + O 9.
    assert_eq!(v.len(), 26);
    s.hold_locked = true;
    v.clear();
    movegen::generate(&s, GenOptions::new(Mode::Sonic, true), &mut v);
    assert_eq!(v.len(), 17);
}

#[test]
fn empty_board_counts() {
    let b = Board::new();
    let want = [(Piece::I, 17), (Piece::O, 9), (Piece::T, 34), (Piece::L, 34), (Piece::J, 34), (Piece::S, 17)];
    for (p, n) in want.into_iter().chain([(Piece::Z, 17)]) {
        for mode in [Mode::Sonic, Mode::Step] {
            let c = movegen::count_from(&b, Falling::spawn(p), Spin::None, mode, false);
            assert_eq!(c, n, "{p:?} {mode:?}");
        }
    }
}

// --- Perft -----------------------------------------------------------------------------------------

const FUSION_QUEUE: [Piece; 7] = [Piece::I, Piece::O, Piece::L, Piece::J, Piece::S, Piece::Z, Piece::T];
const FUSION_COUNTS: [u64; 5] = [17, 153, 5_266, 188_561, 3_500_883];
const FUSION_OPTS: PerftOpts = PerftOpts { mode: Mode::Step, hold: false, by_cells: true };

#[test]
fn perft_fusion_d1_d3() {
    for d in 1..=3 {
        assert_eq!(perft_board(&Board::new(), &FUSION_QUEUE, d, FUSION_OPTS), FUSION_COUNTS[d as usize - 1]);
    }
}

#[test]
#[ignore = "slow in debug; run with --release -- --ignored"]
fn perft_fusion_d4_d5() {
    for d in 4..=5 {
        assert_eq!(perft_board(&Board::new(), &FUSION_QUEUE, d, FUSION_OPTS), FUSION_COUNTS[d as usize - 1]);
    }
}

/// Brute-force bare-board perft with the reference BFS (same semantics as `perft_board`).
fn brute_board(board: &Board, queue: &[Piece], qi: usize, hold: Option<Piece>, depth: u32, o: PerftOpts) -> u64 {
    if depth == 0 {
        return 1;
    }
    let q = |i: usize| queue[i % queue.len()];
    let cur = q(qi);
    let mut branches = vec![(cur, hold, qi + 1, false)];
    if o.hold && hold != Some(cur) {
        branches.push(match hold {
            None => (q(qi + 1), Some(cur), qi + 2, true),
            Some(h) => (h, Some(cur), qi + 1, true),
        });
    }
    let ropts = ReachOpts { das: false, step: o.mode == Mode::Step, by_cells: true };
    let mut n = 0;
    for (piece, nh, nq, flag) in branches {
        let f = Falling::spawn(piece);
        if !f.fits(board) {
            continue;
        }
        let mut seen = BTreeSet::new();
        for r in tetris_sim::reach::reachable_from_with(board, f, Spin::None, flag, ropts) {
            let cells = placement_cells(&r.placement);
            if o.by_cells && !seen.insert(cells) {
                continue;
            }
            let mut b = *board;
            b.place(&cells);
            b.clear_all_lines();
            n += brute_board(&b, queue, nq, nh, depth - 1, o);
        }
    }
    n
}

/// Brute-force state perft with the reference BFS (same semantics as `perft`).
fn brute_state(s: &PlayerState, depth: u32, o: PerftOpts) -> u64 {
    if depth == 0 {
        return 1;
    }
    let ropts = ReachOpts { das: false, step: o.mode == Mode::Step, by_cells: true };
    let mut seen = BTreeSet::new();
    let mut n = 0;
    for r in reachable_with(s, ropts, o.hold) {
        if o.by_cells && !seen.insert((r.placement.hold, placement_cells(&r.placement))) {
            continue;
        }
        if depth == 1 {
            n += 1;
            continue;
        }
        let mut c = s.clone();
        if c.apply_placement(r.placement).topout {
            continue;
        }
        n += brute_state(&c, depth - 1, o);
    }
    n
}

#[test]
fn perft_fast_equals_brute() {
    let mut rng = Lcg(31337);
    let depth = if cfg!(debug_assertions) { 2 } else { 3 };
    for i in 0..6 {
        let board = random_board(&mut rng, STYLES[i % STYLES.len()]);
        let mut queue = Piece::ALL;
        for k in (1..7).rev() {
            queue.swap(k, rng.below(k + 1));
        }
        for mode in [Mode::Sonic, Mode::Step] {
            for hold in [false, true] {
                for by_cells in [false, true] {
                    let o = PerftOpts { mode, hold, by_cells };
                    let fast = perft_board(&board, &queue, depth, o);
                    let slow = brute_board(&board, &queue, 0, None, depth, o);
                    assert_eq!(fast, slow, "board {i} {o:?}\n{board:?}");
                    let s = random_state(&mut rng, board, queue[0]);
                    assert_eq!(perft(&s, depth, o), brute_state(&s, depth, o), "state {i} {o:?}\n{}", describe(&s));
                }
            }
        }
    }
}

#[test]
fn canonical_rotations() {
    let mut v = Vec::new();
    for p in Piece::ALL {
        v.clear();
        movegen::generate_from(&Board::new(), Falling::spawn(p), Spin::None, Mode::Step, false, &mut v);
        for q in &v {
            assert_eq!(q.rot, movegen::canonical_rotation(p, q.rot));
            if matches!(p, Piece::I | Piece::S | Piece::Z) {
                assert!(matches!(q.rot, Rotation::North | Rotation::East));
            }
        }
    }
}
