//! Rules-exact TETR.IO Season 2 (Tetra League) versus simulator core.
//!
//! The rules follow triangle.js (`@haelp/teto` 4.2.8, commit `aa0698ab`) as summarised in
//! `docs/RULES.md`; see `docs/SIM.md` for the module map and API.
//!
//! Hot path: [`PlayerState::apply_placement`] (no allocation unless the incoming garbage queue grows
//! past its inline capacity).

pub mod attack;
pub mod board;
pub mod config;
pub mod falling;
pub mod garbage;
pub mod kicks;
pub mod movegen;
pub mod piece;
pub mod queue;
pub mod reach;
pub mod rng;
pub mod state;
pub mod versus;

#[cfg(feature = "replay")]
pub mod replay;

pub use attack::{AttackResult, ChainState, ClearInfo, Events, attack};
pub use board::Board;
pub use config::Config;
pub use falling::{Falling, Spin};
pub use garbage::{GarbageQueue, IncomingGarbage};
pub use piece::{Piece, Rotation};
pub use queue::{BagSeen, PREVIEW, PieceQueue};
pub use rng::{Rng, SplitMix};
pub use state::{Key, LockObserver, LockOutcome, Placement, PlayerState, Stats};
pub use versus::Match;
