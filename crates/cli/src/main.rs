//! `sim`: the simulator, engine and self-play command-line tools. Run it without arguments for the
//! command list ([`COMMANDS`]).

mod args;
mod bot_cmds;
mod nnue_cmds;
mod selfplay_cmds;
mod sim_cmds;

use std::process::ExitCode;

use args::{Args, CliError, CmdResult};

struct Command {
    name: &'static str,
    args: &'static str,
    about: &'static str,
    run: fn(&Args) -> CmdResult,
}

const COMMANDS: &[Command] = &[
    Command {
        name: "replay",
        args: "<request-or-trace.json> [-o out.json]",
        about: "replay a request, print the trace JSON",
        run: sim_cmds::replay,
    },
    Command {
        name: "diff",
        args: "<trace.json|dir>...",
        about: "replay oracle traces and report the first mismatch",
        run: sim_cmds::diff,
    },
    Command {
        name: "gen",
        args: "<outdir> [--count N] [--seed S] [--pieces P] [--style versus|spin|allclear|topout|long|noise|quad|all]",
        about: "write generated requests",
        run: sim_cmds::gen_requests,
    },
    Command {
        name: "gen-movegen",
        args: "<outdir> [--count N] [--seed S] [--pieces P]",
        about: "requests whose keys come from movegen + pathfinder",
        run: sim_cmds::gen_movegen,
    },
    Command {
        name: "perft",
        args: "--depth N [--seed S] [--mode sonic|step] [--hold] [--by-cells] [--queue IOLJSZT]",
        about: "count placement sequences (docs/MOVEGEN.md)",
        run: sim_cmds::perft,
    },
    Command {
        name: "bench-movegen",
        args: "[--ms M]",
        about: "movegen placements/s on fixed boards + self-play states",
        run: sim_cmds::bench_movegen,
    },
    Command {
        name: "bot-vs-bot",
        args: "--a BOT --b BOT [--games N] [--seed S] [--pps a,b] [--width W] [--width-a W] [--width-b W]\n\
               [--extra-a N] [--extra-b N] [--k-a K] [--k-b K] [--max-pieces P] [--jitter J]\n\
               [--dump-requests DIR [--dump N]] [--log]",
        about: "paired-seed bot match (docs/BOT.md)",
        run: bot_cmds::bot_vs_bot,
    },
    Command {
        name: "weights",
        args: "<preset|weights.json> [-o out.json]",
        about: "print the full weight vector as JSON",
        run: bot_cmds::weights,
    },
    Command {
        name: "tune",
        args: "--config tune.toml [--out DIR] [--generations G] [--max-games N]",
        about: "CMA-ES self-play tuning, resumable (docs/SELFPLAY.md)",
        run: selfplay_cmds::tune,
    },
    Command {
        name: "tournament",
        args: "[--pool DIR] [--bots a,b,..] [--reserve-off a,..] [--pairs N] [--width W] [--seed S]\n\
               [--pps lo,hi] [--varied] [--max-pieces P] [--anchor NAME] [--bootstrap B] [--json OUT]",
        about: "round robin, Bradley-Terry Elo with bootstrap CIs",
        run: selfplay_cmds::tournament,
    },
    Command {
        name: "datagen",
        args: "--bots a,b,.. --games N --out STEM [--width W] [--every K] [--seed S] [--pps lo,hi] [--varied]\n\
               [--max-pieces P] [--batch B] [--siblings K] [--teacher WEIGHTS] [--keep-clears]",
        about: "value-net training data (docs/NNUE.md)",
        run: nnue_cmds::datagen,
    },
    Command {
        name: "nnue-eval",
        args: "--net X.nnue --data STEM [--n N]",
        about: "raw network outputs for the first N samples",
        run: nnue_cmds::nnue_eval,
    },
    Command {
        name: "nnue-random",
        args: "-o X.nnue [--seed S]",
        about: "a random network (pipeline tests, benchmarks)",
        run: nnue_cmds::nnue_random,
    },
];

fn usage() -> ExitCode {
    eprintln!("usage:");
    for c in COMMANDS {
        let indent = " ".repeat(7 + c.name.len());
        eprintln!("  sim {} {}", c.name, c.args.replace('\n', &format!("\n{indent}")));
        eprintln!("      {}", c.about);
    }
    eprintln!("A BOT is a preset (default, steady, greedy-app), a weights .json, or a value network .nnue.");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().and_then(|name| COMMANDS.iter().find(|c| c.name == name)) else { return usage() };
    match (cmd.run)(&Args(&argv[1..])) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(msg)) => {
            eprintln!("sim {}: {msg}", cmd.name);
            usage()
        }
        Err(CliError::Fail(msg)) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}
