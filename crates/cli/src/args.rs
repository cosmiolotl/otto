//! Command-line arguments: flags with values (`--seed 3`), switches (`--hold`) and positionals.

use std::str::FromStr;

use tetris_selfplay::Conditions;

pub enum CliError {
    /// Bad or missing arguments: print the usage.
    Usage(String),
    Fail(String),
}

impl<E: std::fmt::Display> From<E> for CliError {
    fn from(e: E) -> CliError {
        CliError::Fail(e.to_string())
    }
}

pub type CmdResult = Result<(), CliError>;

pub fn usage_error(msg: impl Into<String>) -> CliError {
    CliError::Usage(msg.into())
}

pub struct Args<'a>(pub &'a [String]);

impl<'a> Args<'a> {
    /// The `i`-th argument, if it is not a flag.
    pub fn positional(&self, i: usize) -> Option<&'a str> {
        self.0.get(i).map(String::as_str).filter(|a| !a.starts_with('-'))
    }

    /// Every argument before the first flag.
    pub fn positionals(&self) -> impl Iterator<Item = &'a str> {
        self.0.iter().map(String::as_str).take_while(|a| !a.starts_with('-'))
    }

    /// The value after `name`.
    pub fn get(&self, name: &str) -> Option<&'a str> {
        self.0.iter().position(|a| a == name).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.0.iter().any(|a| a == name)
    }

    pub fn require(&self, name: &str) -> Result<&'a str, CliError> {
        self.get(name).ok_or_else(|| usage_error(format!("missing {name}")))
    }

    /// The value after `name` parsed as a `T`, or `default` if the flag is absent.
    pub fn num<T: FromStr>(&self, name: &str, default: T) -> Result<T, CliError> {
        match self.get(name) {
            None => Ok(default),
            Some(s) => s.parse().map_err(|_| usage_error(format!("{name} {s}: not a valid number"))),
        }
    }

    /// A comma-separated list (empty if the flag is absent).
    pub fn list(&self, name: &str) -> Vec<String> {
        self.get(name)
            .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
            .unwrap_or_default()
    }

    /// `x` or `a,b` (one number means both), or `default` if the flag is absent.
    pub fn pair(&self, name: &str, default: [f64; 2]) -> Result<[f64; 2], CliError> {
        let Some(s) = self.get(name) else { return Ok(default) };
        let v: Vec<f64> =
            s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().map_err(|_| bad_pair(name, s))?;
        match v.as_slice() {
            [x] => Ok([*x, *x]),
            [a, b] => Ok([*a, *b]),
            _ => Err(bad_pair(name, s)),
        }
    }
}

/// Match conditions from `--pps lo,hi` (default `pps`), `--varied` (vary the rules in part of the pairs as
/// [`Conditions::default`] does; otherwise exact TL rules) and `--max-pieces`.
pub fn conditions(args: &Args, pps: [f64; 2]) -> Result<Conditions, CliError> {
    let varied = Conditions::default();
    Ok(Conditions {
        pps: args.pair("--pps", pps)?,
        tl_fraction: if args.has("--varied") { varied.tl_fraction } else { 1.0 },
        max_pieces: args.num("--max-pieces", varied.max_pieces)?,
        ..varied
    })
}

fn bad_pair(name: &str, s: &str) -> CliError {
    usage_error(format!("{name} {s}: expected one number or two separated by a comma"))
}
