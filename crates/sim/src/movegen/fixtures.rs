//! Benchmark positions: fixed boards for movegen, and mid-game states from quick self-play (criterion
//! benches and `sim bench-movegen`).

use super::{GenOptions, Mode, generate};
use crate::board::Board;
use crate::garbage::IncomingGarbage;
use crate::rng::SplitMix;
use crate::state::PlayerState;

/// `(name, board)`: empty, a mid-game ~8-high stack with an overhang and a TSD slot, a messy garbage
/// stack, and a high (~17) stack with wells and holes.
pub fn bench_boards() -> Vec<(&'static str, Board)> {
    vec![
        ("empty", Board::new()),
        (
            "midgame",
            Board::from_ascii(
                "
                #.......##
                ##.....###
                ###.#.####
                ##...#####
                ###.######
                ########.#
                #.########
                ########.#
                ",
            ),
        ),
        (
            "garbage",
            Board::from_ascii(
                "
                ..#.......
                .###...#..
                ####..###.
                GGGGGGGGG.
                GGGG.GGGGG
                GGGG.GGGGG
                G.GGGGGGGG
                GGGGGGGG.G
                GG.GGGGGGG
                GG.GGGGGGG
                GGGGGGG.GG
                ",
            ),
        ),
        (
            "high",
            Board::from_ascii(
                "
                .#........
                ##.....#..
                ###.#..##.
                ###.#####.
                ##..#####.
                ###.######
                ####.#####
                ##...#####
                ###.######
                #######.##
                #.#######.
                ########.#
                ##.######.
                #######.##
                ###.######
                ####.#####
                #####.####
                ",
            ),
        ),
    ]
}

/// Mid-game states (after the 10th placement) from `games` quick solo games of up to `pieces` placements:
/// a cheap low-stack heuristic with some noise, 1-4 lines of incoming garbage every 6 pieces.
pub fn midgame_states(games: i64, pieces: u32) -> Vec<PlayerState> {
    let mut pool = Vec::new();
    let mut buf = Vec::new();
    let mut rng = SplitMix(0x9e37_79b9_7f4a_7c15);
    for g in 0..games {
        let mut s = PlayerState::tl(1000 + g);
        for i in 0..pieces {
            if i % 6 == 0 {
                let amount = rng.below(4) as u32 + 1;
                s.receive_garbage(IncomingGarbage { frame: s.frame, amount, cid: i + 1, gameid: 2 });
            }
            s.tick_n(30);
            if i >= 10 {
                pool.push(s.clone());
            }
            buf.clear();
            generate(&s, GenOptions::new(Mode::Step, true), &mut buf);
            let Some(&p) = buf.iter().min_by_key(|p| {
                let mut c = s.clone();
                let out = c.apply_placement(**p);
                let h = c.board.heights();
                let holes: u32 =
                    (0..10).map(|x| (0..h[x] as usize).filter(|&y| !c.board.get(x, y)).count() as u32).sum();
                let bump: u32 = h.windows(2).map(|w| w[0].abs_diff(w[1]) as u32).sum();
                let top = c.board.top() as u64;
                top * 40 + holes as u64 * 30 + bump as u64 * 4 + out.topout as u64 * 100_000 + rng.below(16)
            }) else {
                break;
            };
            if s.apply_placement(p).topout {
                break;
            }
        }
    }
    pool
}
