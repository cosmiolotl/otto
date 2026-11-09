//! Rule options. [`Config::TL`] is TETR.IO Season 2 Tetra League (RULES.md §1).
//!
//! Player states hold a `&'static Config` so they stay cheap to clone; custom configurations are
//! interned once with [`Config::intern`].

use std::sync::Mutex;

/// `b2bcharging` (`at`, `base`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct B2bCharging {
    pub at: i32,
    pub base: i32,
}

/// `allclear_garbage` / `allclear_b2b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AllClear {
    pub garbage: f64,
    pub b2b: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// `garbagespeed`: frames before an incoming item may enter the board.
    pub garbage_speed: u32,
    /// `garbagecap` (lines per placement), its per-second increase, margin (frames) and max.
    pub garbage_cap: f64,
    pub garbage_cap_increase: f64,
    pub garbage_cap_margin: u32,
    pub garbage_cap_max: f64,
    /// `openerphase`: number of opening placements with doubled cancelling.
    pub opener_phase: u32,
    /// `messiness_change`, `messiness_inner`, `messiness_nosame`, `messiness_center`.
    pub messiness_change: f64,
    pub messiness_within: f64,
    pub messiness_nosame: bool,
    pub messiness_center: bool,
    /// `garbagemultiplier` / `garbageincrease` (per second) / `garbagemargin` (frames).
    pub multiplier_value: f64,
    pub multiplier_increase: f64,
    pub multiplier_margin: u32,
    /// `garbagespecialbonus`.
    pub special_bonus: bool,
    /// `b2bchaining`.
    pub b2b_chaining: bool,
    /// `b2bcharging`; `None` = no surge.
    pub b2b_charging: Option<B2bCharging>,
    /// `allclears`; `None` = no all-clear bonus.
    pub all_clear: Option<AllClear>,
    /// `clutch`.
    pub clutch: bool,
}

impl Config {
    /// TETR.IO Season 2 Tetra League.
    pub const TL: Config = Config {
        garbage_speed: 20,
        garbage_cap: 8.0,
        garbage_cap_increase: 0.0,
        garbage_cap_margin: 0,
        garbage_cap_max: 40.0,
        opener_phase: 14,
        messiness_change: 1.0,
        messiness_within: 0.0,
        messiness_nosame: false,
        messiness_center: false,
        multiplier_value: 1.0,
        multiplier_increase: 0.008,
        multiplier_margin: 10_800,
        special_bonus: true,
        b2b_chaining: false,
        b2b_charging: Some(B2bCharging { at: 4, base: 3 }),
        all_clear: Some(AllClear { garbage: 5.0, b2b: 1 }),
        clutch: true,
    };

    /// The TL config as a `'static` reference.
    pub fn tl() -> &'static Config {
        static TL: Config = Config::TL;
        &TL
    }

    /// Get a `'static` reference to an equal config, leaking at most one allocation per distinct config.
    pub fn intern(self) -> &'static Config {
        if self == Config::TL {
            return Config::tl();
        }
        static POOL: Mutex<Vec<&'static Config>> = Mutex::new(Vec::new());
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = pool.iter().find(|c| ***c == self) {
            return c;
        }
        let c: &'static Config = Box::leak(Box::new(self));
        pool.push(c);
        c
    }
}

impl Default for Config {
    fn default() -> Self {
        Config::TL
    }
}

/// `IncreaseTracker` (`src/engine/utils/increase/index.ts`): `tick(): frame++; if frame > margin: value
/// += increase / 60`. The value is accumulated by repeated f64 addition, as in the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tracker {
    pub value: f64,
}

impl Tracker {
    #[inline]
    pub fn new(base: f64) -> Tracker {
        Tracker { value: base }
    }

    /// Tick for the frame that just became `frame` (the post-increment frame count).
    #[inline]
    pub fn tick(&mut self, frame: u32, increase: f64, margin: u32) {
        if frame > margin {
            self.value += increase / 60.0;
        }
    }
}
