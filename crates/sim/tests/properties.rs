//! Property tests (seeded loops): cell conservation, no overlapping cells, garbage accounting across
//! versus matches, and `apply_placement` ≡ `apply_keys` for every reachable placement.

use tetris_sim::reach::{Reachable, reachable};
use tetris_sim::{IncomingGarbage, Match, MatchResult, PlayerState};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A low-ish random placement (keeps games going for a while).
fn pick(s: &PlayerState, rng: &mut Lcg) -> Reachable {
    let v = reachable(s);
    assert!(!v.is_empty());
    let mut keyed: Vec<(u64, Reachable)> = v
        .into_iter()
        .map(|r| {
            let max_y = r.placement.falling().cells().iter().map(|c| c.1).max().unwrap();
            ((max_y as u64) * 16 + rng.next() % 24, r)
        })
        .collect();
    keyed.sort_by_key(|(k, _)| *k);
    let k = rng.below(keyed.len().min(4));
    keyed.swap_remove(k).1
}

#[test]
fn cell_conservation_and_no_overlap() {
    let mut rng = Lcg(7);
    for game in 0..12 {
        let mut s = PlayerState::tl(500 + game);
        let mut cid = 0;
        for i in 0..120 {
            if i % 4 == 0 {
                cid += 1;
                s.receive_garbage(IncomingGarbage { frame: s.frame, amount: 1 + rng.below(4) as u32, cid, gameid: 2 });
            }
            s.tick_n(25);
            let r = pick(&s, &mut rng);
            let before = s.board;
            let n0 = before.cell_count() as i64;
            let tall = before.top() >= 34;
            let out = s.apply_placement(r.placement);
            // No overlap: every placed cell was empty.
            for &(x, y) in &out.cells {
                assert!(!before.get(x as usize, y as usize), "overlap at ({x},{y})");
            }
            assert!(s.board.check_invariants());
            if !tall {
                let n1 = s.board.cell_count() as i64;
                assert_eq!(n1, n0 + 4 - 10 * out.lines as i64 + 9 * out.tanked as i64, "game {game} piece {i}");
            }
            if out.topout {
                break;
            }
        }
    }
}

#[test]
fn placement_equals_keys() {
    let mut rng = Lcg(99);
    for game in 0..6 {
        let mut s = PlayerState::tl(9000 + game);
        let mut cid = 0;
        for i in 0..40 {
            if i % 6 == 0 {
                cid += 1;
                s.receive_garbage(IncomingGarbage { frame: s.frame, amount: 2, cid, gameid: 2 });
            }
            s.tick_n(30);
            let all = reachable(&s);
            for r in &all {
                assert_eq!(s.check_placement(&r.placement), Ok(()), "{:?}", r.placement);
                let mut a = s.clone();
                let mut b = s.clone();
                let oa = a.apply_placement(r.placement);
                let ob = b.apply_keys(&r.keys).expect("lock");
                assert_eq!(oa, ob, "outcome {:?} keys {:?}", r.placement, r.keys);
                assert_eq!(a, b, "state {:?}", r.placement);
            }
            let r = &all[rng.below(all.len().min(6))];
            if s.apply_placement(r.placement).topout {
                break;
            }
        }
    }
}

#[test]
fn versus_garbage_accounting() {
    let mut rng = Lcg(2024);
    let fpps: [[u32; 2]; 4] = [[30, 30], [12, 40], [45, 9], [20, 21]];
    let mut finished = 0;
    for (g, fpp) in fpps.iter().enumerate() {
        let mut m = Match::tl(77 + g as i64, *fpp);
        let mut n = 0;
        while let Some(p) = m.next_turn() {
            let r = pick(&m.players[p], &mut rng);
            m.play(p, r.placement);
            n += 1;
            // sent by the opponent = landed + cancelled + pending, for both players, at every step.
            for q in 0..2 {
                let recv = &m.players[q];
                let sender = &m.players[1 - q];
                assert_eq!(
                    sender.stats.sent,
                    recv.stats.garbage_landed + recv.stats.garbage_cancelled + recv.garbage.size(),
                    "game {g} after {n} placements, receiver {q}"
                );
                // Own attack spent on cancelling is between 1/2 and 1 of the incoming lines removed.
                let spent = sender.stats.attack - sender.stats.sent;
                assert!(sender.stats.garbage_cancelled >= spent && sender.stats.garbage_cancelled <= 2 * spent);
            }
            // Both clocks in lockstep.
            assert_eq!(m.players[0].frame, m.players[1].frame);
            if n > 600 {
                break;
            }
        }
        if let Some(MatchResult::Win(w)) = m.result {
            assert!(m.players[1 - w].topped_out);
            finished += 1;
        }
    }
    assert!(finished > 0, "no game finished by top-out");
}

#[test]
fn unequal_pps_turn_order() {
    // Player 1 plays 3× faster: in the first 120 frames, player 0 locks at 40/80/120 and player 1 every
    // 13 frames; ties go to player 0.
    let mut m = Match::tl(5, [40, 13]);
    let mut order = Vec::new();
    while m.frame <= 120 {
        let p = m.next_turn().unwrap();
        if m.frame > 120 {
            break;
        }
        order.push((m.frame, p));
        let r = reachable(&m.players[p]).swap_remove(0);
        m.play(p, r.placement);
    }
    let mut expect: Vec<(u32, usize)> = (1..=9).map(|k| (13 * k, 1)).collect();
    expect.extend([(40, 0), (80, 0), (120, 0)]);
    expect.sort_by_key(|&(f, p)| (f, p));
    assert_eq!(order, expect);

    let mut m = Match::tl(5, [20, 20]);
    let a = m.next_turn().unwrap();
    let r = reachable(&m.players[a]).swap_remove(0);
    m.play(a, r.placement);
    let b = m.next_turn().unwrap();
    assert_eq!((a, b), (0, 1));
    assert_eq!(m.frame, 20);
}
