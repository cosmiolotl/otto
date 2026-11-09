//! Per-bot totals over a set of pairs (the numbers RESULTS.md and `sim bot-vs-bot` report).

use serde::Serialize;
use tetris_bot::Mode;

use crate::game::LossCause;
use crate::schedule::PairResult;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BotTotals {
    pub games: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub losses_to_garbage: u64,
    pub losses_stacked_out: u64,
    pub pieces: u64,
    /// Attack generated (pre-cancel).
    pub attack: u64,
    /// Attack sent after own cancelling.
    pub sent: u64,
    /// Of this bot's sent attack: lines the opponent cancelled (opener doubling included) / that landed.
    pub opp_cancelled: u64,
    pub opp_landed: u64,
    pub spikes: u64,
    pub spike_total: u64,
    pub max_spike: u32,
    pub big_spikes: u64,
    /// Moves per controller mode (BUILD, KILL, CANCEL).
    pub modes: [u64; Mode::COUNT],
    pub moves: u64,
    pub think_ns: u64,
}

impl BotTotals {
    pub fn app(&self) -> f64 {
        self.attack as f64 / self.pieces.max(1) as f64
    }
    pub fn avg_spike(&self) -> f64 {
        self.spike_total as f64 / self.spikes.max(1) as f64
    }
    /// Share of this bot's attack that reached the opponent's board: landed / (landed + cancelled).
    pub fn landed_share(&self) -> f64 {
        self.opp_landed as f64 / (self.opp_landed + self.opp_cancelled).max(1) as f64
    }
    pub fn mode_share(&self, m: Mode) -> f64 {
        self.modes[m.index()] as f64 / self.modes.iter().sum::<u64>().max(1) as f64
    }
    pub fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.games.max(1) as f64
    }
    pub fn ms_per_move(&self) -> f64 {
        self.think_ns as f64 / 1e6 / self.moves.max(1) as f64
    }
}

/// Totals per roster index (`n` bots).
pub fn totals(n: usize, pairs: &[PairResult]) -> Vec<BotTotals> {
    let mut t = vec![BotTotals::default(); n];
    for pr in pairs {
        for (g, res) in pr.games.iter().enumerate() {
            // Seat of A is g.
            for (who, seat) in [(pr.a, g), (pr.b, 1 - g)] {
                let x = &mut t[who];
                let (st, opp) = (&res.stats[seat], &res.stats[1 - seat]);
                x.games += 1;
                match res.winner {
                    Some(w) if w == seat => x.wins += 1,
                    Some(_) => {
                        x.losses += 1;
                        match res.loss {
                            Some(LossCause::Garbage) => x.losses_to_garbage += 1,
                            _ => x.losses_stacked_out += 1,
                        }
                    }
                    None => x.draws += 1,
                }
                x.pieces += st.pieces as u64;
                x.attack += st.attack as u64;
                x.sent += st.sent as u64;
                x.opp_cancelled += opp.garbage_cancelled as u64;
                x.opp_landed += opp.garbage_landed as u64;
                let m = &res.metrics[seat];
                x.spikes += m.spikes as u64;
                x.spike_total += m.spike_total as u64;
                x.max_spike = x.max_spike.max(m.max_spike);
                x.big_spikes += m.big_spikes as u64;
                let b = &res.bot[seat];
                for (t, n) in x.modes.iter_mut().zip(b.modes) {
                    *t += n;
                }
                x.moves += b.moves;
                x.think_ns += b.think_ns;
            }
        }
    }
    t
}

/// A fixed-width table of `totals` (one row per bot).
pub fn table(names: &[String], t: &[BotTotals]) -> String {
    let mut s = format!(
        "{:<24} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>7} {:>7} {:>6} {:>6} {:>6} {:>8} {:>7}\n",
        "bot",
        "games",
        "score",
        "APP",
        "spike",
        "maxspk",
        "≥10",
        "landed",
        "cancel",
        "BUILD",
        "KILL",
        "CANCEL",
        "lossGarb",
        "ms/mv"
    );
    for (name, x) in names.iter().zip(t) {
        let cancel = 1.0 - x.landed_share();
        s += &format!(
            "{:<24} {:>6} {:>6.3} {:>6.3} {:>6.2} {:>7} {:>6} {:>6.1}% {:>6.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>7.1}% {:>7.2}\n",
            name,
            x.games,
            x.score(),
            x.app(),
            x.avg_spike(),
            x.max_spike,
            x.big_spikes,
            100.0 * x.landed_share(),
            100.0 * cancel,
            100.0 * x.mode_share(Mode::Build),
            100.0 * x.mode_share(Mode::Kill),
            100.0 * x.mode_share(Mode::Cancel),
            100.0 * x.losses_to_garbage as f64 / x.losses.max(1) as f64,
            x.ms_per_move()
        );
    }
    s
}
