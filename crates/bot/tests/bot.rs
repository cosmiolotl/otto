//! Bot-level tests: information hiding and legality in the real state.

use tetris_bot::{Bot, Observation, SearchOptions, weights};
use tetris_sim::{IncomingGarbage, Match, PREVIEW};

fn opts(width: usize) -> SearchOptions {
    SearchOptions { width, ..Default::default() }
}

/// Play a seeded match for `n` placements with two default bots, returning it.
fn midgame(seed: i64, n: usize) -> Match {
    let w = weights::preset("default").unwrap();
    let mut bots = [Bot::new(w.clone(), opts(8), 1), Bot::new(w, opts(8), 2)];
    let mut m = Match::tl(seed, [24, 26]);
    for _ in 0..n {
        let p = m.next_turn().unwrap();
        let d = bots[p].choose(&Observation::from_match(&m, p)).unwrap();
        m.play(p, d.placement);
        if m.is_over() {
            break;
        }
    }
    m
}

#[test]
fn same_observation_same_move_regardless_of_rngs() {
    for (seed, n) in [(101, 0), (202, 9), (303, 37), (404, 60)] {
        let mut m = midgame(seed, n);
        let p = m.next_turn().unwrap();
        // Some garbage in flight both ways, so the garbage RNG would matter if it leaked.
        m.inject(p, 3);
        m.inject(1 - p, 2);
        let (me, opp) = (&m.players[p], &m.players[1 - p]);
        // Same visible state, different bag RNG / hidden queue tail / garbage RNG.
        let mut me2 = me.clone();
        me2.queue = me.queue.resample_tail(PREVIEW, 0x1234 + seed);
        me2.garbage.reseed(999 + seed);
        let mut opp2 = opp.clone();
        opp2.queue = opp.queue.resample_tail(PREVIEW, 0x777 + seed);
        opp2.garbage.reseed(31 + seed);
        assert_ne!(*me, me2, "the hidden parts really differ");
        let fpp = m.frames_per_piece;
        let o1 = Observation::new(me, opp, fpp[p], fpp[1 - p]);
        let o2 = Observation::new(&me2, &opp2, fpp[p], fpp[1 - p]);
        assert_eq!(o1, o2);
        for name in weights::PRESETS {
            let w = weights::preset(name).unwrap();
            let d1 = Bot::new(w.clone(), opts(24), 5).choose(&o1).unwrap();
            let d2 = Bot::new(w, opts(24), 5).choose(&o2).unwrap();
            assert_eq!(d1, d2, "{name} at seed {seed}");
        }
    }
}

#[test]
fn decision_is_legal_in_the_real_state() {
    let mut m = midgame(55, 30);
    let p = m.next_turn().unwrap();
    m.players[p].receive_garbage(IncomingGarbage { frame: m.frame, amount: 5, cid: 999, gameid: 2 });
    let d = Bot::new(weights::Weights::default(), opts(16), 3).choose(&Observation::from_match(&m, p)).unwrap();
    assert!(m.players[p].check_placement(&d.placement).is_ok());
}
