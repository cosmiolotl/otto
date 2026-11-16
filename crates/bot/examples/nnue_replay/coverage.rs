//! Wider state coverage before replay: all stack heights, pending values, and dense-layer buckets (and, for
//! a `rows-v1-plan` network, every controller mode with varied threat estimates).

use tetris_bot::controller::{Mode, Plan};
use tetris_bot::nnue::features::{PENDING_MAX, extract_static, pending_feature};
use tetris_bot::nnue::{Features, Network, bucket, extract, plan_features};
use tetris_sim::{Board, IncomingGarbage, PlayerState, SplitMix};

pub fn states() -> Vec<PlayerState> {
    let mut rng = SplitMix(91);
    (0..=40)
        .map(|height| {
            let mut state = PlayerState::tl(height + 17);
            let rows: Vec<u16> = (0..height).map(|_| rng.below(1023) as u16 + 1).collect();
            state.board = Board::from_rows(&rows, 0);
            state.chain.b2b = height as i32 - 1;
            state.chain.combo = height as i32 / 2 - 1;
            state.last_was_clear = height % 2 == 0;
            state.frame = height as u32 * 600;
            state.receive_garbage(IncomingGarbage {
                frame: state.frame + 100,
                amount: height as u32,
                cid: 1,
                gameid: 2,
            });
            state
        })
        .collect()
}

pub fn verify(net: &Network) -> usize {
    let states = states();
    let mut checked = 0;
    for (i, state) in states.iter().enumerate() {
        let opponent = &states[(i + 13) % states.len()];
        let fpp = 10 + i as u32;
        let mut table = net.opponent_table(opponent, fpp);
        // Cover every pending value, independently of the original opponent's garbage count.
        table.base_pending = 0;
        let (mut me, mut opp) = (Features::default(), Features::default());
        extract(state, fpp, &mut me);
        let mode = [Mode::Build, Mode::Kill, Mode::Cancel][i % 3];
        let plan = plan_features(&Plan { mode, opp_burst: (i % 15) as f32, opp_fill: (i % 13) as f32 / 12.0 });
        let plan: &[u16] = if net.layout.has_plan() { &plan } else { &[] };
        me.extend(plan);
        extract_static(opponent, fpp, &mut opp);
        let pending_index = opp.len;
        opp.len += 1;
        for pending in 0..=PENDING_MAX {
            opp.idx[pending_index] = pending_feature(pending as u32);
            let mut side = [0; tetris_bot::nnue::net::HALF_MAX];
            net.side(opp.as_slice(), &mut side);
            assert_eq!(table.sides[pending], side, "opponent accumulator mismatch");
            let expected = Network::logit(net.eval_pair(me.as_slice(), opp.as_slice(), bucket(state)));
            let actual = net.evaluate_with(state, fpp, &table, pending as u32, plan);
            assert_eq!(actual.to_bits(), expected.to_bits(), "height {i}, pending {pending}");
            checked += 1;
        }
    }
    checked
}
