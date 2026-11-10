//! WebAssembly bindings for the browser client in `web/`.
//!
//! [`Game`] is a real-time versus match: player 0 is the human (driven key by key from JavaScript, which
//! also runs gravity, lock delay and DAS/ARR), player 1 is Otto (one [`Placement`] per lock, scheduled on
//! the shared frame clock every `otto_fpp` frames).
//!
//! The page runs two copies of the match. The main thread owns the authoritative one and records every
//! state change as an event log ([`Game::take_log`]). A Web Worker replays that log into its own copy
//! ([`Game::apply_log`]) and runs the bot on it ([`Brain::think`]). Both copies run the same deterministic
//! code, so they stay identical; the bot only ever sees an [`Observation`] of the copy.

use tetris_bot::{Bot, HandEval, Observation, SearchOptions};
use tetris_sim::board::{BOARD_HEIGHT, BOARD_WIDTH};
use tetris_sim::versus::route;
use tetris_sim::{Key, LockObserver, LockOutcome, Piece, Placement, PlayerState, Rotation, Spin};
use wasm_bindgen::prelude::*;

/// Rows sent to the renderer (20 visible + 4 of the buffer zone, where spawns and clutch happen).
pub const RENDER_ROWS: usize = 24;

/// Cell codes of [`Game::board`]: 0 empty, 1..=7 a piece (`Piece` index + 1), 8 garbage.
const GARBAGE: u8 = 8;

/// Event kinds in the log; each event is three `u32`s: `[frame, kind, arg]`.
const EV_KEY: u32 = 0;
const EV_GRAVITY_LOCK: u32 = 1;
const EV_OTTO: u32 = 2;

const KEYS: [Key; 11] = Key::ALL;

/// Piece-colour mirror of a board, driven by [`LockObserver`] (the sim keeps only bitboards).
#[derive(Clone)]
struct Colors([[u8; BOARD_WIDTH]; BOARD_HEIGHT]);

impl LockObserver for Colors {
    fn placed(&mut self, piece: Piece, cells: &[(i8, i8); 4]) {
        for &(x, y) in cells {
            if (0..BOARD_WIDTH as i8).contains(&x) && (0..BOARD_HEIGHT as i8).contains(&y) {
                self.0[y as usize][x as usize] = piece.index() as u8 + 1;
            }
        }
    }

    fn cleared(&mut self, mask: u64) {
        let mut w = 0;
        for r in 0..BOARD_HEIGHT {
            if mask >> r & 1 == 0 {
                self.0[w] = self.0[r];
                w += 1;
            }
        }
        for row in &mut self.0[w..] {
            *row = [0; BOARD_WIDTH];
        }
    }

    fn garbage_line(&mut self, column: u8, _cid: u32) {
        // Same row-37 quirk as `Board::insert_garbage`.
        let (r38, r39) = (self.0[38], self.0[39]);
        self.0.copy_within(0..37, 1);
        let mut row = [GARBAGE; BOARD_WIDTH];
        row[column as usize] = 0;
        self.0[0] = row;
        self.0[38] = r38;
        self.0[39] = r39;
    }
}

/// Pack a placement into one `u32` (log events, worker → main thread).
pub fn pack(p: &Placement) -> u32 {
    p.piece.index() as u32
        | (p.hold as u32) << 3
        | (p.rot.index() as u32) << 4
        | ((p.x as i32 + 8) as u32) << 6
        | ((p.y as i32 + 8) as u32) << 11
        | (p.spin as u32) << 17
}

pub fn unpack(v: u32) -> Placement {
    Placement {
        piece: Piece::ALL.into_iter().find(|p| p.index() as u32 == v & 7).unwrap_or(Piece::I),
        hold: v >> 3 & 1 == 1,
        rot: Rotation::from_u8((v >> 4 & 3) as u8),
        x: ((v >> 6 & 31) as i32 - 8) as i8,
        y: ((v >> 11 & 63) as i32 - 8) as i8,
        spin: match v >> 17 & 3 {
            1 => Spin::Mini,
            2 => Spin::Normal,
            _ => Spin::None,
        },
    }
}

#[wasm_bindgen]
pub struct Game {
    players: [PlayerState; 2],
    colors: [Colors; 2],
    frame: u32,
    otto_fpp: u32,
    otto_next: u32,
    last_cid: u32,
    winner: i32,
    record: bool,
    log: Vec<u32>,
    /// Per player: `[seq, piece, lines, spin, all_clear, raw, sent, b2b, combo, broke_b2b, surge, cancelled, tanked]`.
    last_lock: [[i32; 13]; 2],
    lock_seq: i32,
}

#[wasm_bindgen]
impl Game {
    /// Both players share `seed` (TL rule). `record`: keep an event log for a replica (main thread).
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, otto_fpp: u32, record: bool) -> Game {
        let otto_fpp = otto_fpp.max(1);
        Game {
            players: [PlayerState::tl(seed as i64), PlayerState::tl(seed as i64)],
            colors: [Colors([[0; BOARD_WIDTH]; BOARD_HEIGHT]), Colors([[0; BOARD_WIDTH]; BOARD_HEIGHT])],
            frame: 0,
            otto_fpp,
            otto_next: otto_fpp,
            last_cid: 0,
            winner: -1,
            record,
            log: Vec::new(),
            last_lock: [[0; 13]; 2],
            lock_seq: 0,
        }
    }

    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// Advance the shared clock (no-op for frames in the past).
    pub fn advance_to(&mut self, frame: u32) {
        if frame > self.frame {
            let n = frame - self.frame;
            for p in &mut self.players {
                p.tick_n(n);
            }
            self.frame = frame;
        }
    }

    /// -1 while playing, else the winning player.
    pub fn winner(&self) -> i32 {
        self.winner
    }

    // ---- human (player 0) ----

    /// Press one key for the human (`Key::ALL` index). Returns 0 = nothing happened, 1 = moved / rotated /
    /// held, 2 = locked.
    pub fn press(&mut self, key: u8) -> u8 {
        if self.winner >= 0 || key as usize >= KEYS.len() {
            return 0;
        }
        // A press that did nothing (blocked shift or rotation, refused hold) changed no state, so the
        // replica does not need it: held keys re-pressed every frame would otherwise flood the log.
        let frame = self.frame;
        let r = self.press_inner(KEYS[key as usize]);
        if r > 0 && self.record {
            self.log.extend([frame, EV_KEY, key as u32]);
        }
        r
    }

    /// Lock the human's piece where it rests (lock delay ran out: `#lock(false)`).
    pub fn gravity_lock(&mut self) {
        if self.winner >= 0 {
            return;
        }
        self.push_log(EV_GRAVITY_LOCK, 0);
        self.gravity_lock_inner();
    }

    /// Whether the human's piece is resting on something.
    pub fn grounded(&self) -> bool {
        let s = &self.players[0];
        s.falling.is_grounded(&s.board)
    }

    /// Row of the human's piece (`falling.y`), for lock-reset bookkeeping.
    pub fn falling_y(&self) -> i32 {
        self.players[0].falling.y as i32
    }

    // ---- Otto (player 1) ----

    /// Frame at which Otto's next piece is due.
    pub fn otto_due(&self) -> u32 {
        self.otto_next
    }

    /// Change Otto's speed; takes effect from its next lock (and pulls a far-away lock closer).
    pub fn set_otto_fpp(&mut self, fpp: u32) {
        self.otto_fpp = fpp.max(1);
        self.otto_next = self.otto_next.min(self.frame + self.otto_fpp);
    }

    /// Lock Otto's packed placement now. Returns false (and does nothing) if it is not legal here.
    pub fn play_otto(&mut self, packed: u32) -> bool {
        if self.winner >= 0 {
            return false;
        }
        let p = unpack(packed);
        if self.players[1].check_placement(&p).is_err() {
            return false;
        }
        self.push_log(EV_OTTO, packed);
        self.play_otto_inner(p);
        true
    }

    // ---- replica ----

    /// Drain the event log (`[frame, kind, arg]` triples).
    pub fn take_log(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.log)
    }

    /// Replay events recorded by another `Game` with the same seed.
    pub fn apply_log(&mut self, events: &[u32]) {
        for e in events.chunks_exact(3) {
            self.advance_to(e[0]);
            match e[1] {
                EV_KEY => {
                    if let Some(&k) = KEYS.get(e[2] as usize) {
                        self.press_inner(k);
                    }
                }
                EV_GRAVITY_LOCK => self.gravity_lock_inner(),
                EV_OTTO => self.play_otto_inner(unpack(e[2])),
                _ => {}
            }
        }
    }

    // ---- rendering ----

    /// `RENDER_ROWS × 10` cell codes, bottom row first.
    pub fn board(&self, p: usize) -> Vec<u8> {
        self.colors[p & 1].0[..RENDER_ROWS].iter().flatten().copied().collect()
    }

    /// `[piece, rot, x0, y0, … x3, y3, ghost x0, y0, … x3, y3, hold (-1), hold_locked, next ×5, b2b, combo,
    /// pieces, lines, attack, sent, topped_out, last_spin]`.
    pub fn info(&self, p: usize) -> Vec<i32> {
        let s = &self.players[p & 1];
        let mut v = Vec::with_capacity(40);
        v.push(s.falling.piece.index() as i32);
        v.push(s.falling.rot.index() as i32);
        for (x, y) in s.falling.cells() {
            v.extend([x as i32, y as i32]);
        }
        let mut ghost = s.falling;
        ghost.sonic_drop(&s.board);
        for (x, y) in ghost.cells() {
            v.extend([x as i32, y as i32]);
        }
        v.push(s.hold.map_or(-1, |h| h.index() as i32));
        v.push(s.hold_locked as i32);
        for i in 0..5 {
            v.push(s.queue.peek(i).index() as i32);
        }
        v.extend([s.chain.b2b, s.chain.combo]);
        v.extend([s.stats.pieces, s.stats.lines, s.stats.attack, s.stats.sent].map(|n| n as i32));
        v.push(s.topped_out as i32);
        v.push(s.last_spin as i32);
        v
    }

    /// Incoming garbage: `[amount, frame it may enter the board]` per item, oldest first.
    pub fn incoming(&self, p: usize) -> Vec<i32> {
        let s = &self.players[p & 1];
        s.garbage.items.iter().flat_map(|g| [g.amount as i32, g.ready_at(s.cfg) as i32]).collect()
    }

    /// The player's last lock (see the field docs); `seq` grows with every lock of either player.
    pub fn last_lock(&self, p: usize) -> Vec<i32> {
        self.last_lock[p & 1].to_vec()
    }
}

impl Game {
    fn push_log(&mut self, kind: u32, arg: u32) {
        if self.record {
            self.log.extend([self.frame, kind, arg]);
        }
    }

    fn press_inner(&mut self, key: Key) -> u8 {
        let (moved, out) = self.players[0].press_with(key, &mut self.colors[0]);
        match out {
            Some(out) => {
                self.after_lock(0, &out);
                2
            }
            None => moved as u8,
        }
    }

    fn gravity_lock_inner(&mut self) {
        let out = self.players[0].lock_with(false, &mut self.colors[0]);
        self.after_lock(0, &out);
    }

    fn play_otto_inner(&mut self, p: Placement) {
        // `PlayerState::apply_placement`, with the colour observer.
        let s = &mut self.players[1];
        if p.hold {
            s.hold();
        }
        s.falling = p.falling();
        s.last_spin = p.spin;
        let out = s.lock_with(true, &mut self.colors[1]);
        self.otto_next = self.frame + self.otto_fpp;
        self.after_lock(1, &out);
    }

    fn after_lock(&mut self, p: usize, out: &LockOutcome) {
        route(&mut self.players, p, out.sent.as_slice(), &mut self.last_cid);
        let chain = self.players[p].chain;
        self.lock_seq += 1;
        self.last_lock[p] = [
            self.lock_seq,
            out.piece.index() as i32,
            out.lines as i32,
            out.spin as i32,
            out.all_clear as i32,
            out.raw.total() as i32,
            out.sent.total() as i32,
            chain.b2b,
            chain.combo,
            out.broke_b2b as i32,
            out.surge as i32,
            out.cancelled as i32,
            out.tanked as i32,
        ];
        if out.topout && self.winner < 0 {
            self.winner = 1 - p as i32;
        }
    }
}

/// Otto's weights: `mean-0511`, the final mean of the M5 CMA-ES tune (generation 511,
/// `runs/aws/runs/m5/mean.json`).
const WEIGHTS: &str = include_str!("../../../weights/mean-0511.json");

/// Otto's search: the "d10" configuration, 6 beam plies (current + preview) plus 4 greedy plies on a
/// sampled bag-consistent continuation (`bot-vs-bot --extra 4`, K = 1).
pub const EXTRA_PLIES: usize = 4;
pub const CONTINUATIONS: usize = 1;

/// Otto: the bot, run on a replica [`Game`] (in the worker).
#[wasm_bindgen]
pub struct Brain {
    bot: Bot<HandEval>,
}

#[wasm_bindgen]
impl Brain {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, seed: u32) -> Brain {
        let opts = SearchOptions {
            width: width.max(1) as usize,
            extra_plies: EXTRA_PLIES,
            continuations: CONTINUATIONS,
            ..SearchOptions::default()
        };
        let w: tetris_bot::Weights = serde_json::from_str(WEIGHTS).expect("weights/mean-0511.json");
        Brain { bot: Bot::new(w, opts, seed as u64) }
    }

    /// Name of the weights in use.
    pub fn weights_name(&self) -> String {
        self.bot.eval.weights.name.clone()
    }

    pub fn set_width(&mut self, width: u32) {
        self.bot.opts.width = width.max(1) as usize;
    }

    /// Choose Otto's next placement in `game`. `human_fpp`: the human's recent frames per piece (the
    /// controller's threat model uses the opponent's speed). Returns `[packed placement, mode, score,
    /// nodes]`, or an empty array if there is no legal placement.
    pub fn think(&mut self, game: &Game, human_fpp: u32) -> Vec<f64> {
        let obs = Observation::new(&game.players[1], &game.players[0], game.otto_fpp, human_fpp.max(1));
        match self.bot.choose(&obs) {
            Some(d) => vec![pack(&d.placement) as f64, d.mode.index() as f64, d.score as f64, d.nodes as f64],
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        for piece in Piece::ALL {
            for rot in Rotation::ALL {
                for (x, y) in [(-2, 0), (0, 1), (9, 39), (4, 22)] {
                    for spin in [Spin::None, Spin::Mini, Spin::Normal] {
                        for hold in [false, true] {
                            let p = Placement { piece, hold, rot, x, y, spin };
                            assert_eq!(unpack(pack(&p)), p);
                        }
                    }
                }
            }
        }
    }

    /// A recorded match replayed into a second copy ends in the identical state, and the bot's placements
    /// are legal on the authoritative copy.
    #[test]
    fn replica_stays_in_sync() {
        let mut main = Game::new(12345, 20, true);
        let mut replica = Game::new(12345, 20, false);
        let mut brain = Brain::new(16, 7);
        let mut human = Brain::new(8, 9);
        let mut f = 0;
        while main.winner() < 0 && f < 60 * 60 {
            f += 1;
            main.advance_to(f);
            // Human: every 25 frames, lock a bot-chosen placement through keys (hold + move + rotate + drop).
            if f % 25 == 0 {
                let obs = Observation::new(&main.players[0], &main.players[1], 25, 20);
                if let Some(d) = human.bot.choose(&obs) {
                    let p = d.placement;
                    if p.hold {
                        main.press(8);
                    }
                    // Rotate in place, then shift; good enough to exercise the log (not always reachable).
                    for _ in 0..p.rot.index() {
                        main.press(5);
                    }
                    let dx = p.x - main.players[0].falling.x;
                    for _ in 0..dx.unsigned_abs() {
                        main.press(if dx < 0 { 0 } else { 1 });
                    }
                    main.press(10); // one gravity step
                    if f % 50 == 0 {
                        main.press(4);
                        main.gravity_lock();
                    } else {
                        main.press(9);
                    }
                }
            }
            if f >= main.otto_due() && main.winner() < 0 {
                replica.apply_log(&main.take_log());
                replica.advance_to(f);
                let r = brain.think(&replica, 25);
                assert!(!r.is_empty());
                assert!(main.play_otto(r[0] as u32), "bot placement illegal on the main copy");
            }
        }
        replica.apply_log(&main.take_log());
        replica.advance_to(main.frame());
        let (h, o) = (main.players[0].stats, main.players[1].stats);
        assert!(h.pieces >= 20 && o.pieces >= 20, "too short to mean anything: {h:?} {o:?}");
        assert_eq!(main.players, replica.players);
        assert_eq!(main.winner(), replica.winner());
        assert_eq!(main.board(0), replica.board(0));
        assert_eq!(main.board(1), replica.board(1));
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;

    #[test]
    fn embedded_weights_are_mean_0511() {
        let b = Brain::new(96, 1);
        assert_eq!(b.weights_name(), "mean-0511");
        assert_eq!(b.bot.opts.depth + b.bot.opts.extra_plies, 10);
    }

    /// Native think time per move at full strength (`cargo test --release -p otto-wasm -- --nocapture
    /// think_time`); the browser is roughly 1.5–2× slower.
    #[test]
    fn think_time() {
        let mut g = Game::new(0x5eed, 12, false);
        let mut b = Brain::new(96, 1);
        let t0 = std::time::Instant::now();
        let mut n = 0;
        while n < 60 && g.winner() < 0 {
            g.advance_to(g.otto_due());
            let r = b.think(&g, 30);
            assert!(!r.is_empty() && g.play_otto(r[0] as u32));
            n += 1;
        }
        println!("{n} moves, {:.2} ms/move", t0.elapsed().as_secs_f64() * 1000.0 / n as f64);
    }
}
