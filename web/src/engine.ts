// Loads the WebAssembly engine (built by `npm run wasm` into src/wasm/). Used by the page and the worker.
import init, { Brain, Game } from './wasm/otto_wasm.js';
import wasmUrl from './wasm/otto_wasm_bg.wasm?url';

let ready: Promise<unknown> | null = null;

export function loadEngine(): Promise<unknown> {
  ready ??= init({ module_or_path: wasmUrl });
  return ready;
}

export { Brain, Game, wasmUrl };

/** `Key::ALL` indices of the sim (crates/sim/src/state.rs). */
export const Key = {
  MoveLeft: 0,
  MoveRight: 1,
  DasLeft: 2,
  DasRight: 3,
  SoftDrop: 4,
  RotateCw: 5,
  RotateCcw: 6,
  Rotate180: 7,
  Hold: 8,
  HardDrop: 9,
  SoftDropStep: 10,
} as const;

/** `Game::press` results. */
export const Press = { None: 0, Moved: 1, Locked: 2 } as const;

/** Piece indices of the sim (`Piece` discriminants) → letter. */
export const PIECES = ['I', 'O', 'T', 'L', 'J', 'S', 'Z'] as const;

/** Layout of `Game::info`. */
export const Info = {
  piece: 0,
  rot: 1,
  cells: 2, // 4 × (x, y)
  ghost: 10, // 4 × (x, y)
  hold: 18,
  holdLocked: 19,
  next: 20, // 5
  b2b: 25,
  combo: 26,
  pieces: 27,
  lines: 28,
  attack: 29,
  sent: 30,
  toppedOut: 31,
  lastSpin: 32,
} as const;

/** Layout of `Game::last_lock`. */
export const Lock = {
  seq: 0,
  piece: 1,
  lines: 2,
  spin: 3,
  allClear: 4,
  raw: 5,
  sent: 6,
  b2b: 7,
  combo: 8,
  brokeB2b: 9,
  surge: 10,
  cancelled: 11,
  tanked: 12,
} as const;

export const RENDER_ROWS = 24;
export const FPS = 60;
