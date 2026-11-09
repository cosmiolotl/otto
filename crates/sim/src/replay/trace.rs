//! Replaying a request into a trace (`docs/TRACE_FORMAT.md` "Trace").

use serde_json::{Value, json};

use super::ReplayError;
use super::colors::ColorBoard;
use super::request::{Request, Step};
use crate::garbage::IncomingGarbage;
use crate::state::{Key, LockOutcome, PlayerState};
use crate::versus::route;

/// Number of upcoming pieces in `PlayerState.next`.
pub const NEXT_LEN: usize = 10;

/// `PlayerState` JSON for one player.
pub fn player_json(s: &PlayerState, colors: &ColorBoard) -> Value {
    debug_assert_eq!(&colors.bits(), s.board.rows(), "colour board out of sync with bitboard");
    let piece = |p: crate::Piece| Value::String(p.to_char().to_string());
    json!({
        "frame": s.frame,
        "board": colors.rows(),
        "current": piece(s.falling.piece),
        "hold": s.hold.map_or(Value::Null, piece),
        "hold_locked": s.hold_locked,
        "next": s.queue.iter(NEXT_LEN).map(piece).collect::<Vec<_>>(),
        "b2b": s.chain.b2b,
        "combo": s.chain.combo,
        "garbage_queue": s.garbage.items.iter().map(|g| json!({
            "amount": g.amount, "frame": g.frame, "cid": g.cid, "gameid": g.gameid
        })).collect::<Vec<_>>(),
        "garbage_rng": s.garbage.rng.state(),
        "last_column": s.garbage.last_column,
        "has_changed_column": s.garbage.has_changed_column,
        "garbage_sent": s.garbage.sent,
        "multiplier": s.multiplier.value,
        "stats": {
            "pieces": s.stats.pieces,
            "lines": s.stats.lines,
            "attack": s.stats.attack,
            "sent": s.stats.sent,
            "cleared": s.stats.cleared,
        },
        "topped_out": topped_out(s),
    })
}

/// `topped_out` as the oracle reports it: `engine.toppedOut` (the falling piece does not fit).
fn topped_out(s: &PlayerState) -> bool {
    s.falling_blocked()
}

fn lock_json(out: &LockOutcome, added: &[(u8, u32)]) -> Value {
    let mut cells: Vec<(i8, i8)> = out.cells.to_vec();
    cells.sort_by_key(|&(x, y)| (y, x));
    json!({
        "piece": out.piece.to_char().to_string(),
        "cells": cells.iter().map(|&(x, y)| json!([x, y])).collect::<Vec<_>>(),
        "lines": out.lines,
        "spin": out.spin.as_str(),
        "garbage_cleared": out.garbage_cleared,
        "raw_garbage": out.raw.as_slice(),
        "garbage": out.sent.as_slice(),
        "surge": out.surge,
        "garbage_added": added.iter().map(|&(c, cid)| json!({"column": c, "cid": cid})).collect::<Vec<_>>(),
        "topout": out.topout,
    })
}

/// Replay a parsed request. `raw` is echoed verbatim as `trace.request`.
pub fn replay_request(req: &Request, raw: &Value) -> Result<Value, ReplayError> {
    if req.format != 1 {
        return Err(ReplayError::Invalid(format!("format {} (expected 1)", req.format)));
    }
    if req.players == 0 {
        return Err(ReplayError::Invalid("players must be >= 1".into()));
    }
    let cfg = req.config.clone().unwrap_or_default().to_config()?.intern();
    let mut players: Vec<PlayerState> = (0..req.players).map(|_| PlayerState::new(req.seed, cfg)).collect();
    let mut colors: Vec<ColorBoard> = (0..req.players).map(|_| ColorBoard::default()).collect();
    let mut last_cid: u32 = 0;
    let snapshot = |players: &[PlayerState], colors: &[ColorBoard]| -> Vec<Value> {
        players.iter().zip(colors).map(|(p, c)| player_json(p, c)).collect()
    };

    let initial = snapshot(&players, &colors);
    let mut placements = Vec::new();

    for (si, step) in req.steps.iter().enumerate() {
        match step {
            Step::Tick { tick } => {
                for p in &mut players {
                    p.tick_n(*tick);
                }
            }
            Step::Garbage { player, garbage_in } => {
                let p = players
                    .get_mut(*player)
                    .ok_or_else(|| ReplayError::Invalid(format!("step {si}: player {player}")))?;
                for g in garbage_in {
                    last_cid += 1;
                    if g.amount > 0 {
                        p.receive_garbage(IncomingGarbage {
                            frame: p.frame,
                            amount: g.amount as u32,
                            cid: last_cid,
                            gameid: 0,
                        });
                    }
                }
            }
            Step::Keys { player, keys } => {
                let pi = *player;
                if pi >= players.len() {
                    return Err(ReplayError::Invalid(format!("step {si}: player {pi}")));
                }
                let keys: Vec<Key> = keys
                    .iter()
                    .map(|k| {
                        Key::from_name(k).ok_or_else(|| ReplayError::Invalid(format!("step {si}: unknown key `{k}`")))
                    })
                    .collect::<Result<_, _>>()?;
                if keys.last() != Some(&Key::HardDrop) || keys[..keys.len() - 1].contains(&Key::HardDrop) {
                    return Err(ReplayError::Invalid(format!("step {si}: keys must end with exactly one hardDrop")));
                }
                let out = players[pi].apply_keys_with(&keys, &mut colors[pi]).expect("ends with hardDrop");
                let added = colors[pi].take_added();
                route(&mut players, pi, out.sent.as_slice(), &mut last_cid);
                placements.push(json!({
                    "step": si,
                    "player": pi,
                    "frame": players[pi].frame,
                    "lock": lock_json(&out, &added),
                    "after": snapshot(&players, &colors),
                }));
            }
        }
    }

    Ok(json!({
        "format": 1,
        "request": raw,
        "simulator": { "name": "tetris-sim", "version": env!("CARGO_PKG_VERSION") },
        "initial": initial,
        "placements": placements,
    }))
}

/// Replay a request given as JSON (a bare request, or a trace containing `request`).
pub fn replay_request_value(v: &Value) -> Result<Value, ReplayError> {
    let raw = v.get("request").unwrap_or(v);
    let req: Request = serde_json::from_value(raw.clone())?;
    replay_request(&req, raw)
}
