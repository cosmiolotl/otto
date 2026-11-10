# web: play against Otto in the browser

A static site. The engine (`crates/sim` + `crates/bot`) is compiled to WebAssembly by `engine/` and runs
entirely in the visitor's browser: no server, no account.

```sh
cd web
npm install
npm run wasm      # build engine/ → src/wasm/ (needs the wasm32 target and wasm-bindgen-cli, see below)
npm run dev       # http://localhost:5173
npm run build     # wasm + typecheck + dist/ (deploy dist/ to any static host)
```

One-time toolchain setup:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked   # must equal the version pinned in engine/Cargo.toml
```

## How it fits together

| Piece | Where | What it does |
|---|---|---|
| `engine/` (`otto-wasm`) | Rust → `src/wasm/` | `Game`: a real-time versus match (human = player 0, key by key; Otto = player 1, one placement every `60 / pps` frames) with a piece-colour board; `Brain`: the bot (weights `weights/mean-0511.json`, the M5 tune's final mean; beam width 96, depth 6 + 4 greedy plies = "d10"). |
| `src/main.ts` | page | The authoritative `Game`, a 60 Hz frame clock from `performance.now()`, rounds and the session score (not persisted). |
| `src/input.ts` | page | Your piece: DAS / ARR / SDF / DCD, TL gravity (0.02 G, rising after 2:00), lock delay (31 grounded frames, 15 resets). |
| `src/worker.ts` | Web Worker | A replica `Game` rebuilt from the page's event log, and Otto's search on it. Benchmarks the device on start. |
| `src/ui.ts`, `src/render.ts` | page | Player clusters (canvas boards, hold / next, B2B, stats) and the settings panel. |

**Why a replica.** Otto's search must not block your input, so it runs in a worker. Instead of shipping
game state across threads, the page records every state change (`Game::take_log`: your effective key
presses, gravity locks, Otto's placements, each with its frame) and the worker replays them into its own
copy (`Game::apply_log`). Both copies run the same deterministic code, so they are identical; the
`replica_stays_in_sync` test in `engine/` checks this over a whole game. A few frames before Otto's piece is
due, the page asks the worker for a decision on the replica as of that frame, then locks it on time.

**Speed limit.** Otto's speed is a pure pieces-per-second setting (0.5–10). On start the worker times 24
moves at full search width. The panel shows the speed this device can sustain
(`1000 / (1.25 × p90 think ms + 4)`), and warns when the setting is above it: past that point Otto's moves
arrive late, so it plays slower than the setting.

Design: the Paper file "Otto — Web Play" (direction A2, Studio). Icons: [Lucide](https://lucide.dev).
