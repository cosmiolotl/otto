//! A bot as rosters, matches and the tuner pass it around: a name, search options and an evaluator.

use std::sync::Arc;

use crate::bot::Bot;
use crate::eval::{AnyEval, HandEval, NnueEval};
use crate::nnue::Network;
use crate::search::SearchOptions;
use crate::weights::{self, Weights};

#[derive(Clone, Debug)]
pub enum EvalSpec {
    Hand(Weights),
    /// A network and its optional release-controller weights ([`NnueEval::with_controller`]). Equality is
    /// identity of the shared network (and equal controller weights).
    Nnue(Arc<Network>, Option<Weights>),
}

impl PartialEq for EvalSpec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (EvalSpec::Hand(a), EvalSpec::Hand(b)) => a == b,
            (EvalSpec::Nnue(a, wa), EvalSpec::Nnue(b, wb)) => Arc::ptr_eq(a, b) && wa == wb,
            _ => false,
        }
    }
}

impl EvalSpec {
    /// A fresh evaluator of this spec (call [`Evaluator::begin`](crate::Evaluator::begin) before evaluating).
    pub fn make_eval(&self) -> AnyEval {
        match self {
            EvalSpec::Hand(w) => AnyEval::Hand(Box::new(HandEval::new(w.clone()))),
            EvalSpec::Nnue(net, w) => AnyEval::Nnue(Box::new(NnueEval::with_controller(net.clone(), w.clone()))),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BotSpec {
    pub name: String,
    pub opts: SearchOptions,
    pub eval: EvalSpec,
}

impl BotSpec {
    pub fn hand(weights: Weights, opts: SearchOptions) -> BotSpec {
        BotSpec { name: weights.name.clone(), opts, eval: EvalSpec::Hand(weights) }
    }

    /// A network bot without a controller.
    pub fn nnue(name: String, net: Arc<Network>, opts: SearchOptions) -> BotSpec {
        BotSpec::nnue_with_controller(name, net, None, opts)
    }

    /// A network bot whose release controller uses `controller`'s thresholds (see [`NnueEval`]).
    pub fn nnue_with_controller(
        name: String,
        net: Arc<Network>,
        controller: Option<Weights>,
        opts: SearchOptions,
    ) -> BotSpec {
        BotSpec { name, opts, eval: EvalSpec::Nnue(net, controller) }
    }

    /// - `NET.nnue`: a network bot, named after the file. A `rows-v1-plan` network needs controller weights,
    ///   so this is an error for one.
    /// - `NET.nnue@WEIGHTS` (a weights file or a preset): a network bot whose release controller uses those
    ///   weights, named `NET@WEIGHTS-NAME`. A `rows-v1-plan` network reads the plan; for a `rows-v1` network
    ///   the controller only labels its moves (mode statistics), the evaluation is unchanged.
    /// - anything else: a weight preset or a weights file ([`weights::load`]).
    pub fn load(spec: &str, opts: SearchOptions) -> Result<BotSpec, String> {
        let (net_path, controller) = match spec.rsplit_once(".nnue@") {
            Some((stem, w)) => (format!("{stem}.nnue"), Some(weights::load(w)?)),
            None if spec.ends_with(".nnue") => (spec.to_string(), None),
            None => return Ok(BotSpec::hand(weights::load(spec)?, opts)),
        };
        let net = Network::load(&net_path).map_err(|e| e.to_string())?;
        if net.layout.has_plan() && controller.is_none() {
            return Err(format!(
                "{net_path}: a {} network reads the release controller's plan; give it controller weights as \
                 {net_path}@WEIGHTS (a weights file or a preset, e.g. the teacher's)",
                net.layout.name()
            ));
        }
        let mut name = weights::file_stem(&net_path);
        if let Some(w) = &controller {
            name = format!("{name}@{}", w.name);
        }
        Ok(BotSpec::nnue_with_controller(name, Arc::new(net), controller, opts))
    }

    /// The hand-written weights, for a hand bot.
    pub fn weights(&self) -> Option<&Weights> {
        match &self.eval {
            EvalSpec::Hand(w) => Some(w),
            EvalSpec::Nnue(..) => None,
        }
    }

    pub fn make_bot(&self, seed: u64) -> Bot<AnyEval> {
        Bot::with_eval(self.eval.make_eval(), self.opts, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::Layout;

    fn temp_net(layout: Layout) -> String {
        let dir = std::env::temp_dir().join(format!("otto-spec-{}-{}", std::process::id(), layout.name()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("net.nnue");
        Network::random_layout(1, layout, 128).save(&path).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn network_specs() {
        let opts = SearchOptions::default();
        let rows = temp_net(Layout::RowsV1);
        let plan = temp_net(Layout::RowsV1Plan);
        let b = BotSpec::load(&rows, opts).unwrap();
        assert!(matches!(&b.eval, EvalSpec::Nnue(n, None) if n.layout == Layout::RowsV1));
        assert_eq!(b.name, "net");
        let b = BotSpec::load(&format!("{rows}@default"), opts).unwrap();
        assert!(matches!(&b.eval, EvalSpec::Nnue(_, Some(w)) if w.name == "default"));
        assert_eq!(b.name, "net@default");
        let err = BotSpec::load(&plan, opts).unwrap_err();
        assert!(err.contains("@WEIGHTS"), "{err}");
        let b = BotSpec::load(&format!("{plan}@default"), opts).unwrap();
        assert!(matches!(&b.eval, EvalSpec::Nnue(n, Some(_)) if n.layout == Layout::RowsV1Plan));
        assert!(BotSpec::load(&format!("{plan}@no-such-preset"), opts).is_err());
        std::fs::remove_file(rows).ok();
        std::fs::remove_file(plan).ok();
    }
}
