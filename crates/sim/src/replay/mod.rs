//! Oracle trace replayer and diff (`docs/TRACE_FORMAT.md`). Behind the `replay` feature; the simulator
//! core never touches serde.
//!
//! - [`replay_request`]: run a request and produce a trace (`serde_json::Value`) in the oracle's shape.
//! - [`diff_traces`] / [`diff_trace_file`]: replay a golden trace's request and report the first
//!   mismatching field.

mod colors;
mod diff;
pub mod generate;
mod request;
mod trace;

pub use colors::ColorBoard;
pub use diff::{DiffReport, Mismatch, diff_trace_file, diff_traces, diff_values, trace_files};
pub use request::{Request, RequestBuilder, RequestConfig, Step};
pub use trace::{player_json, replay_request, replay_request_value};

/// Errors from parsing or replaying a request.
#[derive(Debug)]
pub enum ReplayError {
    Json(serde_json::Error),
    Io(std::io::Error),
    Invalid(String),
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::Json(e) => write!(f, "json: {e}"),
            ReplayError::Io(e) => write!(f, "io: {e}"),
            ReplayError::Invalid(s) => write!(f, "invalid request: {s}"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<serde_json::Error> for ReplayError {
    fn from(e: serde_json::Error) -> Self {
        ReplayError::Json(e)
    }
}

impl From<std::io::Error> for ReplayError {
    fn from(e: std::io::Error) -> Self {
        ReplayError::Io(e)
    }
}
