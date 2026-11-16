// Messages between the page (authoritative match) and the worker (replica + bot).
import type { Engine } from './settings';

export type ToWorker =
  /** Play with this engine from now on. `netUrl`: where the network file is (for `nnue`). */
  | { t: 'engine'; engine: Engine; netUrl: string; netName: string }
  /** New round: a fresh replica with this seed. */
  | { t: 'reset'; seed: number; fpp: number }
  /** Events recorded by the page's `Game` since the last batch, in order. */
  | { t: 'log'; events: Uint32Array }
  /** Choose Otto's next placement on the replica as of `frame`. */
  | { t: 'think'; id: number; round: number; frame: number; humanFpp: number };

export type FromWorker =
  /** Network download progress, in bytes (`total` 0 when unknown). */
  | { t: 'progress'; loaded: number; total: number }
  /** Otto now plays with `engine` (bot `name`); its benchmarked think time per move on this device. */
  | { t: 'engine'; engine: Engine; name: string; thinkMs: number }
  /** The network could not be loaded; Otto keeps (or falls back to) the hand-written engine. */
  | { t: 'engineError'; message: string }
  /** `packed` = -1: no legal placement (Otto is topped out). */
  | { t: 'decision'; id: number; round: number; packed: number; mode: number; nodes: number; ms: number };
