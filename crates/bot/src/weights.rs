//! The bot's parameters: one flat, named, serde-serialisable vector (`weights/*.json`).
//!
//! Every evaluation weight and every controller threshold is an `f32` field, so the M5 CMA-ES tuner
//! (`crates/selfplay`) can treat the whole thing as a vector ([`Weights::to_vec`] /
//! [`Weights::set_from_vec`], with [`Weights::BOUNDS`]).
//! Only `name` and the `controller` switch are outside the vector.
//!
//! Units: scores are roughly "lines of attack". A feature is measured in its natural unit (cells, rows,
//! lines, b2b levels), so a weight of -1.0 on `holes` means "a hole costs about one line of attack".

use serde::{Deserialize, Serialize};

macro_rules! weights {
    ($( $(#[doc = $doc:literal])* $name:ident = $default:expr, [$lo:expr, $hi:expr]; )*) => {
        /// Evaluation weights and controller parameters (see the module docs and `docs/BOT.md`).
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        #[serde(default, deny_unknown_fields)]
        pub struct Weights {
            /// Preset or file name (informational). Empty when a file has none.
            #[serde(default)]
            pub name: String,
            /// Release controller on (KILL / CANCEL / BUILD) or off (always BUILD).
            pub controller: bool,
            $( $(#[doc = $doc])* pub $name: f32, )*
        }

        impl Weights {
            /// Names of the vector entries, in vector order.
            pub const NAMES: &'static [&'static str] = &[$(stringify!($name)),*];
            /// `(lo, hi)` search bounds per vector entry (for the tuner's normalised space).
            pub const BOUNDS: &'static [(f32, f32)] = &[$(($lo, $hi)),*];
            pub const LEN: usize = Self::NAMES.len();

            /// The tunable entries as a flat vector, in [`Self::NAMES`] order.
            pub fn to_vec(&self) -> Vec<f32> {
                vec![$(self.$name),*]
            }

            /// Overwrite the tunable entries from a flat vector (length [`Self::LEN`]).
            pub fn set_from_vec(&mut self, v: &[f32]) {
                assert_eq!(v.len(), Self::LEN, "weight vector length");
                let mut it = v.iter().copied();
                $( self.$name = it.next().unwrap(); )*
            }
        }

        impl Default for Weights {
            /// The default bot (controller on, reserve features on).
            fn default() -> Weights {
                Weights { name: "default".into(), controller: true, $( $name: $default, )* }
            }
        }
    };
}

weights! {
    // ---- Shape ----
    /// Empty cells with a filled cell above them in the same column.
    holes = -1.6, [-6.0, 0.0];
    /// Rows that contain at least one hole.
    hole_rows = -0.9, [-6.0, 0.0];
    /// Filled cells above a hole (digging cost).
    covered = -0.12, [-2.0, 0.0];
    /// Sum of |height differences| of neighbouring columns (the well column excluded).
    bump = -0.12, [-2.0, 0.0];
    /// Sum of squared height differences (the well column excluded).
    bump_sq = -0.035, [-1.0, 0.0];
    /// Highest column.
    max_height = -0.08, [-2.0, 0.5];
    /// Mean column height.
    avg_height = -0.12, [-2.0, 0.5];
    /// BUILD height cap (rows): above it, `over_cap` applies per row squared.
    height_cap = 9.0, [4.0, 18.0];
    over_cap = -0.35, [-4.0, 0.0];
    /// Row transitions (filled/empty changes along rows, walls count as filled), below the stack top.
    row_trans = -0.07, [-1.0, 0.0];
    /// Column transitions (filled/empty changes up the columns, the floor counts as filled).
    col_trans = -0.1, [-1.0, 0.0];
    /// Depth of the deepest well (rows, capped at `well_cap`).
    well_depth = 0.12, [-1.0, 1.0];
    well_cap = 6.0, [1.0, 12.0];
    /// Well in an edge column (0 or 9).
    well_edge = 0.15, [-2.0, 2.0];
    // ---- Setups ----
    /// Best T-spin double slot: 0.5 for the shape plus 1 per line that a T-spin would clear now.
    tslot = 2.0, [-1.0, 5.0];
    /// Rows above the well bottom that are full except the well column (capped at 4).
    quad_ready = 0.3, [-1.0, 3.0];
    /// Low (≤ 4 rows) hole-free board with even cell count: all-clear potential.
    pc_ready = 0.3, [-2.0, 4.0];
    // ---- Chain ----
    /// B2B chain active (`b2b >= 0`).
    b2b_active = 3.0, [-2.0, 6.0];
    /// Per b2b level (`b2b + 1`), up to `reserve_target`. Surge charge equals the counter once it is ≥ 4.
    b2b_level = 1.0, [-1.0, 3.0];
    /// Reserve target (b2b levels); levels above it earn `b2b_excess` each.
    reserve_target = 10.0, [4.0, 20.0];
    b2b_excess = 0.3, [-1.0, 3.0];
    /// Combo counter (`combo + 1` while a combo is running).
    combo = 0.15, [-1.0, 3.0];
    // ---- Accumulated along the search path (discounted by `discount` per ply) ----
    /// Attack generated (pre-cancel `raw`, includes released surge).
    attack = 1.0, [0.0, 6.0];
    /// Attack sent to the opponent (post-cancel).
    sent = 0.2, [-2.0, 4.0];
    /// Incoming lines cancelled.
    cancelled = 0.5, [-2.0, 4.0];
    /// Incoming lines that entered the board.
    tanked = -0.15, [-4.0, 0.0];
    /// B2B chains broken (per break, any level ≥ 0).
    b2b_break = -3.0, [-8.0, 0.0];
    /// Lines cleared for no attack and no garbage rows (burning the stack).
    waste = -1.0, [-3.0, 0.0];
    /// T pieces locked without a T-spin clear (a T is the best spin piece).
    t_waste = 0.0, [-4.0, 0.0];
    /// Per-ply discount of path terms (earlier attack is worth more).
    discount = 0.92, [0.5, 1.0];
    // ---- Garbage state ----
    /// Pending incoming lines at the node.
    pending = -0.2, [-3.0, 0.0];
    /// `danger_sq · max(0, max_height + pending + burst_danger · opp_burst - danger_level)²`.
    danger_level = 13.0, [6.0, 20.0];
    danger_sq = -0.3, [-4.0, 0.0];
    // ---- Controller ----
    /// KILL if `my_burst - my_pending + opp_pending - opp_absorb > kill_threshold` (lines).
    kill_threshold = 0.0, [-10.0, 10.0];
    /// Scale of the opponent's burst in `opp_absorb` (they cancel with it) and in `danger`.
    opp_burst_scale = 0.7, [0.0, 2.0];
    /// CANCEL if `sigmoid((my_pending + opp_burst_scale · opp_burst - my_margin) / 3) > cancel_threshold`.
    cancel_threshold = 0.5, [0.05, 0.95];
    /// Extra weight on path attack in KILL mode.
    kill_attack = 2.0, [0.0, 8.0];
    /// Extra weight on path cancelled lines in CANCEL mode (and on tanked lines, negated).
    cancel_bonus = 1.0, [0.0, 6.0];
    /// Attack is worth `1 + pressure · opp_fill` times more when the opponent is high (controller only).
    pressure = 0.0, [0.0, 4.0];
    /// The opponent's burst estimate counts this much towards `danger_sq` (controller only).
    burst_danger = 1.0, [0.0, 2.0];
}

impl Weights {
    /// The same weights with the reserve logic off (burst vs steady in `docs/RESULTS.md`): controller off, no
    /// B2B/reserve terms, attack rewarded once sent. Only the reserve logic changes.
    pub fn reserve_off(&self) -> Weights {
        Weights {
            name: format!("{}-steady", self.name),
            controller: false,
            b2b_active: 0.0,
            b2b_level: 0.0,
            b2b_excess: 0.0,
            b2b_break: 0.0,
            sent: 0.4,
            ..self.clone()
        }
    }
}

/// A named preset: `"default"`, `"steady"`, `"greedy-app"`.
pub fn preset(name: &str) -> Option<Weights> {
    let default = Weights::default();
    Some(match name {
        "default" => default,
        "steady" => Weights { name: "steady".into(), ..default.reserve_off() },
        // Maximise attack per piece over the search horizon (no discount, no reserve, no controller).
        "greedy-app" => Weights {
            name: "greedy-app".into(),
            sent: 0.0,
            cancelled: 0.0,
            attack: 4.0,
            discount: 1.0,
            ..default.reserve_off()
        },
        _ => return None,
    })
}

pub const PRESETS: [&str; 3] = ["default", "steady", "greedy-app"];

/// A preset name or a path to a `weights/*.json` file.
pub fn load(spec: &str) -> Result<Weights, String> {
    if let Some(w) = preset(spec) {
        return Ok(w);
    }
    let text = std::fs::read_to_string(spec).map_err(|e| format!("{spec}: {e}"))?;
    let mut w: Weights = serde_json::from_str(&text).map_err(|e| format!("{spec}: {e}"))?;
    if w.name.is_empty() {
        w.name = file_stem(spec);
    }
    Ok(w)
}

/// A file's name without directory and extension (the name of a bot loaded from it).
pub(crate) fn file_stem(path: &str) -> String {
    std::path::Path::new(path).file_stem().map_or(path.into(), |s| s.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip_and_bounds() {
        let w = Weights::default();
        let v = w.to_vec();
        assert_eq!(v.len(), Weights::LEN);
        assert_eq!(Weights::BOUNDS.len(), Weights::LEN);
        for (i, (&x, &(lo, hi))) in v.iter().zip(Weights::BOUNDS).enumerate() {
            assert!(lo < hi && (lo..=hi).contains(&x), "{} = {x} outside [{lo}, {hi}]", Weights::NAMES[i]);
        }
        let mut u = Weights::default();
        let doubled: Vec<f32> = v.iter().map(|x| x * 2.0).collect();
        u.set_from_vec(&doubled);
        assert_eq!(u.holes, w.holes * 2.0);
        u.set_from_vec(&v);
        assert_eq!(u, w);
    }

    #[test]
    fn json_roundtrip_and_partial_files() {
        for name in PRESETS {
            let w = preset(name).unwrap();
            let s = serde_json::to_string_pretty(&w).unwrap();
            let back: Weights = serde_json::from_str(&s).unwrap();
            assert_eq!(back, w);
        }
        // Missing fields take the defaults; unknown fields are rejected.
        let w: Weights = serde_json::from_str(r#"{"holes": -3.0}"#).unwrap();
        assert_eq!(w.holes, -3.0);
        assert_eq!(w.bump, Weights::default().bump);
        assert!(serde_json::from_str::<Weights>(r#"{"holez": -3.0}"#).is_err());
    }
}
