//! One self-play game: the only place a bot-vs-bot game is played.
//!
//! A game is a pure function of its [`GameSpec`] (match seed, bot per seat, frames per piece per seat, bot
//! RNG seeds, piece limit, config). Every placement is checked against `check_placement` and movegen's
//! output for the **real** state before it is played, and at the end the garbage accounting identity
//! (sent = landed + cancelled + pending, per receiver) is asserted. Optionally the game is recorded as an
//! oracle request (pathfinder keys), which replays to the same game in the simulator and in triangle.js.

use serde::{Deserialize, Serialize};
use tetris_bot::{BotSpec, BotStats, Decision, Mode, Observation};
use tetris_sim::movegen::{self, GenOptions, Pathfinder};
use tetris_sim::replay::{Request, RequestBuilder};
use tetris_sim::{Config, Match, MatchResult, Stats};

/// Frames between two attacking placements that still count as one spike (1 s at 60 fps).
pub const SPIKE_WINDOW: u32 = 60;

#[derive(Clone, Debug)]
pub struct GameSpec {
    pub seed: i64,
    pub bots: [BotSpec; 2],
    pub fpp: [u32; 2],
    pub bot_seeds: [u64; 2],
    /// Total placements (both players) before the game is a draw.
    pub max_pieces: u32,
    /// Record the game as an oracle request ([`GameResult::request`]).
    pub record_keys: bool,
    pub cfg: &'static Config,
}

/// Why the loser lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossCause {
    /// Garbage rose into the board on the topping-out lock: killed by attack.
    Garbage,
    /// No garbage entered on the fatal lock: the stack itself reached the spawn (garbage that landed
    /// earlier may still have helped).
    Stack,
}

/// Per-seat metrics that [`tetris_sim::Stats`] does not have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatMetrics {
    /// Spikes: maximal runs of attacking placements (pre-cancel attack > 0) where each one locks within
    /// [`SPIKE_WINDOW`] frames of the previous attacking one. Size = total attack of the run.
    pub spikes: u32,
    pub spike_total: u32,
    pub max_spike: u32,
    /// Spikes of at least 10 lines.
    pub big_spikes: u32,
}

impl SeatMetrics {
    fn close(&mut self, run: u32) {
        if run > 0 {
            self.spikes += 1;
            self.spike_total += run;
            self.max_spike = self.max_spike.max(run);
            self.big_spikes += (run >= 10) as u32;
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameResult {
    pub seed: i64,
    pub fpp: [u32; 2],
    /// `None` = draw (piece limit).
    pub winner: Option<usize>,
    /// Set when there is a winner.
    pub loss: Option<LossCause>,
    pub frames: u32,
    pub stats: [Stats; 2],
    pub bot: [BotStats; 2],
    pub metrics: [SeatMetrics; 2],
    /// Controller mode of every move, per seat.
    pub mode_log: [Vec<Mode>; 2],
    /// The game as an oracle request (`docs/TRACE_FORMAT.md`), if recorded.
    pub request: Option<Request>,
}

pub fn play_game(spec: &GameSpec) -> GameResult {
    play_game_with(spec, |_, _, _| {})
}

/// [`play_game`], calling `on_move(match, player, decision)` before each placement is played.
pub fn play_game_with(spec: &GameSpec, mut on_move: impl FnMut(&Match, usize, &Decision)) -> GameResult {
    let mut m = Match::new(spec.seed, spec.cfg, spec.fpp);
    let mut bots = [0, 1].map(|i| spec.bots[i].make_bot(spec.bot_seeds[i]));
    let mut buf = Vec::with_capacity(128);
    let mut recording = spec.record_keys.then(|| (Pathfinder::new(), RequestBuilder::new(spec.seed, 2)));
    let mut keys = Vec::new();
    let mut mode_log: [Vec<Mode>; 2] = [Vec::new(), Vec::new()];
    let mut metrics = [SeatMetrics::default(); 2];
    // Current spike run per seat: (attack so far, frame of the last attacking lock).
    let mut run = [(0u32, 0u32); 2];
    let mut loss = None;
    let mut n = 0;
    while n < spec.max_pieces {
        let Some(p) = m.next_turn() else { break };
        let d = bots[p]
            .choose(&Observation::from_match(&m, p))
            .unwrap_or_else(|| panic!("seed {} player {p}: no placement", spec.seed));
        let real = &m.players[p];
        if let Err(e) = real.check_placement(&d.placement) {
            panic!(
                "seed {} player {p} piece {}: illegal placement {:?}: {e:?}",
                spec.seed, real.stats.pieces, d.placement
            );
        }
        let gen_mode = spec.bots[p].opts.gen_mode;
        buf.clear();
        movegen::generate(real, GenOptions::new(gen_mode, true), &mut buf);
        assert!(buf.contains(&d.placement), "seed {} player {p}: placement not in movegen output", spec.seed);
        if let Some((pf, req)) = recording.as_mut() {
            keys.clear();
            assert!(pf.find(real, &d.placement, gen_mode, &mut keys), "no path");
            req.tick_to(m.frame);
            req.keys(p, &keys);
        }
        on_move(&m, p, &d);
        mode_log[p].push(d.mode);
        let out = m.play(p, d.placement);
        let atk = out.raw.total();
        if atk > 0 {
            let (size, at) = run[p];
            if size > 0 && m.frame - at > SPIKE_WINDOW {
                metrics[p].close(size);
                run[p] = (atk, m.frame);
            } else {
                run[p] = (size + atk, m.frame);
            }
        }
        if out.topout {
            loss = Some(if out.tanked > 0 { LossCause::Garbage } else { LossCause::Stack });
        }
        n += 1;
    }
    for p in 0..2 {
        metrics[p].close(run[p].0);
    }
    for q in 0..2 {
        let (recv, sender) = (&m.players[q], &m.players[1 - q]);
        assert_eq!(
            sender.stats.sent,
            recv.stats.garbage_landed + recv.stats.garbage_cancelled + recv.garbage.size(),
            "seed {}: garbage accounting for receiver {q}",
            spec.seed
        );
    }
    GameResult {
        seed: spec.seed,
        fpp: spec.fpp,
        winner: match m.result {
            Some(MatchResult::Win(w)) => Some(w),
            _ => None,
        },
        loss,
        frames: m.frame,
        stats: [m.players[0].stats, m.players[1].stats],
        bot: [bots[0].stats, bots[1].stats],
        metrics,
        mode_log,
        request: recording.map(|(_, req)| req.finish()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Conditions;
    use tetris_bot::{SearchOptions, weights};
    use tetris_sim::SplitMix;

    fn spec(seed: i64, cfg: &'static Config, max_pieces: u32, record_keys: bool) -> GameSpec {
        let opts = SearchOptions { width: 8, ..Default::default() };
        GameSpec {
            seed,
            bots: [
                BotSpec::hand(weights::preset("default").unwrap(), opts),
                BotSpec::hand(weights::preset("greedy-app").unwrap(), opts),
            ],
            fpp: [24, 25],
            bot_seeds: [1, 2],
            max_pieces,
            record_keys,
            cfg,
        }
    }

    #[test]
    fn deterministic_and_consistent() {
        let s = spec(4242, Config::tl(), 400, false);
        let (a, b) = (play_game(&s), play_game(&s));
        assert_eq!(a.stats, b.stats);
        assert_eq!(a.metrics, b.metrics);
        assert_eq!(a.winner, b.winner);
        assert_eq!(a.mode_log, b.mode_log);
        assert_eq!(a.loss.is_some(), a.winner.is_some());
        for p in 0..2 {
            let m = &a.metrics[p];
            assert_eq!(m.spike_total, a.stats[p].attack, "every attack line belongs to one spike");
            assert!(m.max_spike <= m.spike_total && m.big_spikes <= m.spikes);
        }
    }

    #[test]
    fn varied_rules_play_a_legal_game() {
        let (_, cfg) = Conditions { tl_fraction: 0.0, ..Default::default() }.sample(&mut SplitMix(3));
        assert!(play_game(&spec(99, cfg, 200, false)).stats[0].pieces > 50);
    }

    /// Replaying the recorded request in the simulator reproduces the game (the oracle check does the same
    /// with triangle.js; see docs/BOT.md).
    #[test]
    fn recorded_request_replays_the_game() {
        let g = play_game(&spec(77, Config::tl(), 160, true));
        let req = serde_json::to_value(g.request.as_ref().unwrap()).unwrap();
        let placements = g.stats[0].pieces + g.stats[1].pieces;
        let steps = req["steps"].as_array().unwrap();
        assert_eq!(steps.iter().filter(|s| s.get("keys").is_some()).count() as u32, placements);
        let trace = tetris_sim::replay::replay_request_value(&req).unwrap();
        let last = trace["placements"].as_array().unwrap().last().unwrap();
        for p in 0..2 {
            let st = &last["after"][p]["stats"];
            assert_eq!(st["pieces"].as_u64().unwrap(), g.stats[p].pieces as u64);
            assert_eq!(st["attack"].as_u64().unwrap(), g.stats[p].attack as u64);
            assert_eq!(st["sent"].as_u64().unwrap(), g.stats[p].sent as u64);
        }
    }
}
