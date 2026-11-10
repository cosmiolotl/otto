// Messages between the page (authoritative match) and the worker (replica + bot).

export type ToWorker =
  /** New round: a fresh replica with this seed. */
  | { t: 'reset'; seed: number; fpp: number }
  /** Events recorded by the page's `Game` since the last batch, in order. */
  | { t: 'log'; events: Uint32Array }
  /** Choose Otto's next placement on the replica as of `frame`. */
  | { t: 'think'; id: number; round: number; frame: number; humanFpp: number };

export type FromWorker =
  /** Engine loaded and benchmarked: think time per move on this device. */
  | { t: 'ready'; thinkMs: number }
  /** `packed` = -1: no legal placement (Otto is topped out). */
  | { t: 'decision'; id: number; round: number; packed: number; mode: number; nodes: number; ms: number };
