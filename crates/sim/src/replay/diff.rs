//! Minimal readable diff between an oracle trace and the simulator's replay of its request.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::ReplayError;
use super::trace::replay_request_value;

/// Fields compared with a tolerance instead of exactly (TRACE_FORMAT.md "Comparison rules").
const TOLERANT: &[(&str, f64)] = &[("multiplier", 1e-9)];

#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    /// JSON path, e.g. `placements[3].after[1].garbage_queue[0].amount`.
    pub path: String,
    pub expected: Value,
    pub actual: Value,
    /// Index into `request.steps` (`None` for `initial` or structural mismatches).
    pub step: Option<u64>,
    /// Index into `placements` (`None` for `initial`).
    pub placement: Option<usize>,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let short = |v: &Value| {
            let s = v.to_string();
            if s.len() > 300 { format!("{}…", &s[..300]) } else { s }
        };
        write!(f, "{}", self.path)?;
        if let Some(s) = self.step {
            write!(f, " (step {s})")?;
        }
        write!(f, "\n    oracle: {}\n    sim:    {}", short(&self.expected), short(&self.actual))
    }
}

#[derive(Clone, Debug)]
pub struct DiffReport {
    pub name: String,
    /// Placements in the oracle trace.
    pub placements: usize,
    /// Placements that matched before the first mismatch (all of them if `mismatch` is `None`).
    pub matched: usize,
    pub mismatch: Option<Mismatch>,
}

impl DiffReport {
    pub fn ok(&self) -> bool {
        self.mismatch.is_none()
    }
}

fn num_eq(key: Option<&str>, a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => {
            if let Some(&(_, tol)) = key.and_then(|k| TOLERANT.iter().find(|(n, _)| *n == k)) {
                (x - y).abs() < tol
            } else {
                // Exact; compare as integers when both are integers to avoid f64 precision issues.
                match (a.as_i64(), b.as_i64()) {
                    (Some(i), Some(j)) => i == j,
                    _ => x == y,
                }
            }
        }
        _ => false,
    }
}

/// First difference between two JSON values (`expected` = oracle). `key` is the field name of this value.
pub fn diff_values(path: &str, key: Option<&str>, expected: &Value, actual: &Value) -> Option<(String, Value, Value)> {
    match (expected, actual) {
        (Value::Number(_), Value::Number(_)) => {
            if num_eq(key, expected, actual) {
                None
            } else {
                Some((path.to_string(), expected.clone(), actual.clone()))
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if let Some(d) = diff_values(&format!("{path}[{i}]"), key, x, y) {
                    return Some(d);
                }
            }
            if a.len() != b.len() {
                // Report the whole arrays: short and most readable.
                return Some((path.to_string(), expected.clone(), actual.clone()));
            }
            None
        }
        (Value::Object(a), Value::Object(b)) => {
            for (k, x) in a {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match b.get(k) {
                    Some(y) => {
                        if let Some(d) = diff_values(&p, Some(k), x, y) {
                            return Some(d);
                        }
                    }
                    None => return Some((p, x.clone(), Value::String("<missing>".into()))),
                }
            }
            for (k, y) in b {
                if !a.contains_key(k) {
                    let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                    return Some((p, Value::String("<missing>".into()), y.clone()));
                }
            }
            None
        }
        _ => {
            if expected == actual {
                None
            } else {
                Some((path.to_string(), expected.clone(), actual.clone()))
            }
        }
    }
}

/// Compare an oracle trace with a simulator trace (both full trace objects).
pub fn diff_traces(name: &str, oracle: &Value, sim: &Value) -> DiffReport {
    let empty = Vec::new();
    let op = oracle.get("placements").and_then(Value::as_array).unwrap_or(&empty);
    let sp = sim.get("placements").and_then(Value::as_array).unwrap_or(&empty);
    let mk = |m: Option<Mismatch>, matched| DiffReport {
        name: name.to_string(),
        placements: op.len(),
        matched,
        mismatch: m,
    };

    let oi = oracle.get("initial").unwrap_or(&Value::Null);
    let si = sim.get("initial").unwrap_or(&Value::Null);
    if let Some((path, e, a)) = diff_values("initial", None, oi, si) {
        return mk(Some(Mismatch { path, expected: e, actual: a, step: None, placement: None }), 0);
    }
    for (i, (o, s)) in op.iter().zip(sp).enumerate() {
        if let Some((path, e, a)) = diff_values(&format!("placements[{i}]"), None, o, s) {
            let step = o.get("step").and_then(Value::as_u64);
            return mk(Some(Mismatch { path, expected: e, actual: a, step, placement: Some(i) }), i);
        }
    }
    if op.len() != sp.len() {
        let m = Mismatch {
            path: "placements.length".into(),
            expected: Value::from(op.len()),
            actual: Value::from(sp.len()),
            step: None,
            placement: Some(op.len().min(sp.len())),
        };
        return mk(Some(m), op.len().min(sp.len()));
    }
    mk(None, op.len())
}

/// Load an oracle trace, replay its request, and diff.
pub fn diff_trace_file(path: &Path) -> Result<DiffReport, ReplayError> {
    let text = std::fs::read_to_string(path)?;
    let oracle: Value = serde_json::from_str(&text)?;
    let sim = replay_request_value(&oracle)?;
    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    Ok(diff_traces(&name, &oracle, &sim))
}

/// All `*.json` files in `dir`, sorted.
pub fn trace_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().is_some_and(|e| e == "json") {
            v.push(p);
        }
    }
    v.sort();
    Ok(v)
}
