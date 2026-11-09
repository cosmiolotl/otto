//! Differential tests: replay oracle traces and require zero mismatches.
//!
//! - `traces/golden/*.json` (written by the oracle harness in `oracle/`). Skipped with a message if the
//!   directory is missing or empty.
//! - `crates/sim/tests/data/selfcheck/*.json`: a few short engine traces produced during M2 with the
//!   stand-in runner (`scratch/m2/run_trace.ts`), so the diff path is always exercised.
//! - `SIM_TRACE_DIR=<dir>`: an extra directory (e.g. a large corpus in `traces/corpus/`).

use std::path::{Path, PathBuf};

use tetris_sim::replay::{diff_trace_file, trace_files};

fn check_dir(dir: &Path, label: &str) -> Option<(usize, usize)> {
    let files = match trace_files(dir) {
        Ok(f) if !f.is_empty() => f,
        Ok(_) => {
            eprintln!("[{label}] SKIPPED: {} contains no *.json traces", dir.display());
            return None;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("[{label}] SKIPPED: {} does not exist", dir.display());
            return None;
        }
        Err(e) => panic!("[{label}] {}: {e}", dir.display()),
    };
    let mut failures = Vec::new();
    let mut placements = 0;
    for f in &files {
        match diff_trace_file(f) {
            Ok(r) => {
                placements += r.placements;
                if let Some(m) = &r.mismatch {
                    failures.push(format!(
                        "{} (placement {:?}, {} matched before):\n  {m}",
                        r.name, m.placement, r.matched
                    ));
                }
            }
            Err(e) => failures.push(format!("{}: {e}", f.display())),
        }
    }
    eprintln!("[{label}] {}/{} traces, {placements} placements checked", files.len() - failures.len(), files.len());
    assert!(failures.is_empty(), "[{label}] {} trace(s) mismatch:\n{}", failures.len(), failures.join("\n"));
    Some((files.len(), placements))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn golden_traces_match() {
    check_dir(&repo_root().join("traces/golden"), "traces/golden");
}

#[test]
fn selfcheck_traces_match() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/selfcheck");
    let (n, p) = check_dir(&dir, "selfcheck").expect("selfcheck traces are checked in");
    assert!(n >= 5 && p > 100);
}

#[test]
fn extra_trace_dir_matches() {
    if let Ok(d) = std::env::var("SIM_TRACE_DIR") {
        check_dir(Path::new(&d), "SIM_TRACE_DIR");
    }
}
