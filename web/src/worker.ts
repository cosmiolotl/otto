/// <reference lib="webworker" />
// Otto's brain. Keeps a replica of the match by replaying the page's event log, and runs the bot on it.
import { Brain, Game, loadEngine } from './engine';
import type { FromWorker, ToWorker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

/** Beam width: Otto's full strength (the width mean-0511 was tuned at). Depth is fixed in the engine (d10). */
export const WIDTH = 96;

let replica: Game | null = null;
let brain: Brain | null = null;
const queue: ToWorker[] = [];
let booted = false;

function post(m: FromWorker): void {
  self.postMessage(m);
}

/** Time `n` moves of self-play from a fresh game: the device's think time per move, in ms (p90). */
function bench(n: number): number {
  const g = new Game(0x5eed, 12, false);
  const b = new Brain(WIDTH, 1);
  const times: number[] = [];
  for (let i = 0; i < n && g.winner() < 0; i++) {
    g.advance_to(g.otto_due());
    const t0 = performance.now();
    const r = b.think(g, 30);
    times.push(performance.now() - t0);
    if (r.length === 0 || !g.play_otto(r[0])) break;
  }
  g.free();
  b.free();
  times.sort((a, b) => a - b);
  // Skip the first (cold) moves' outliers by using the 90th percentile of the warm ones.
  return times[Math.min(times.length - 1, Math.floor(times.length * 0.9))] ?? 10;
}

function handle(m: ToWorker): void {
  switch (m.t) {
    case 'reset':
      replica?.free();
      brain?.free();
      replica = new Game(m.seed, m.fpp, false);
      brain = new Brain(WIDTH, m.seed ^ 0x9e3779b9);
      break;
    case 'log':
      replica?.apply_log(m.events);
      break;
    case 'think': {
      if (!replica || !brain) return;
      replica.advance_to(m.frame);
      const t0 = performance.now();
      const r = brain.think(replica, m.humanFpp);
      const ms = performance.now() - t0;
      if (r.length === 0) post({ t: 'decision', id: m.id, round: m.round, packed: -1, mode: 0, nodes: 0, ms });
      else post({ t: 'decision', id: m.id, round: m.round, packed: r[0], mode: r[1], nodes: r[3], ms });
      break;
    }
  }
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  if (!booted) queue.push(e.data);
  else handle(e.data);
};

loadEngine().then(() => {
  const ms = bench(24);
  booted = true;
  post({ t: 'ready', thinkMs: ms });
  for (const m of queue.splice(0)) handle(m);
});
