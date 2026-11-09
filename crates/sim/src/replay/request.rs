//! Request JSON (`docs/TRACE_FORMAT.md` "Request").

use serde::{Deserialize, Serialize};

use super::ReplayError;
use crate::config::Config;
use crate::state::Key;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Request {
    pub format: u32,
    pub seed: i64,
    pub players: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<RequestConfig>,
    pub steps: Vec<Step>,
}

/// Optional overrides of the TL S2 defaults. Unknown keys are kept in `extra` and rejected by
/// [`RequestConfig::to_config`], so a new oracle option can never be silently ignored.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RequestConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gravity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub garbage_speed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub garbage_cap: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opener_phase: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messiness_change: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messiness_within: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier_increase: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier_margin_frames: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl RequestConfig {
    pub fn to_config(&self) -> Result<Config, ReplayError> {
        if let Some(k) = self.extra.keys().next() {
            return Err(ReplayError::Invalid(format!("unsupported config key `{k}`")));
        }
        if let Some(g) = self.gravity
            && g != 0.0
        {
            return Err(ReplayError::Invalid(format!("gravity {g}: the simulator has no gravity (DECISIONS.md D3)")));
        }
        let mut c = Config::TL;
        if let Some(v) = self.garbage_speed {
            c.garbage_speed = v;
        }
        if let Some(v) = self.garbage_cap {
            c.garbage_cap = v;
        }
        if let Some(v) = self.opener_phase {
            c.opener_phase = v;
        }
        if let Some(v) = self.messiness_change {
            c.messiness_change = v;
        }
        if let Some(v) = self.messiness_within {
            c.messiness_within = v;
        }
        if let Some(v) = self.multiplier_value {
            c.multiplier_value = v;
        }
        if let Some(v) = self.multiplier_increase {
            c.multiplier_increase = v;
        }
        if let Some(v) = self.multiplier_margin_frames {
            c.multiplier_margin = v;
        }
        Ok(c)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GarbageIn {
    pub amount: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Step {
    Tick { tick: u32 },
    Keys { player: usize, keys: Vec<String> },
    Garbage { player: usize, garbage_in: Vec<GarbageIn> },
}

/// Builds a request step by step (format 1, no gravity), tracking the frame the tick steps have reached.
#[derive(Clone, Debug)]
pub struct RequestBuilder {
    req: Request,
    frame: u32,
}

impl RequestBuilder {
    pub fn new(seed: i64, players: usize) -> RequestBuilder {
        let config = RequestConfig { gravity: Some(0.0), ..Default::default() };
        RequestBuilder { req: Request { format: 1, seed, players, config: Some(config), steps: Vec::new() }, frame: 0 }
    }

    /// The frame all players are at after the steps so far.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// A tick step up to `frame` (nothing if the request is already there).
    pub fn tick_to(&mut self, frame: u32) {
        if frame > self.frame {
            self.req.steps.push(Step::Tick { tick: frame - self.frame });
            self.frame = frame;
        }
    }

    pub fn keys(&mut self, player: usize, keys: &[Key]) {
        self.req.steps.push(Step::Keys { player, keys: keys.iter().map(|k| k.name().to_string()).collect() });
    }

    /// External garbage for `player`, one item per amount.
    pub fn garbage(&mut self, player: usize, amounts: &[u32]) {
        let garbage_in = amounts.iter().map(|&a| GarbageIn { amount: a as i64 }).collect();
        self.req.steps.push(Step::Garbage { player, garbage_in });
    }

    pub fn finish(self) -> Request {
        self.req
    }
}
