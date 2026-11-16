/// <reference lib="webworker" />
// Otto's brain. Keeps a replica of the match by replaying the page's event log, and runs the bot on it.
// The bot is the hand-written engine (built into the wasm) or the value network, which is downloaded the
// first time the player picks it.
import { Brain, Game, Net, loadEngine } from './engine';
import type { FromWorker, ToWorker } from './protocol';
import type { Engine } from './settings';

declare const self: DedicatedWorkerGlobalScope;

/** Beam width: Otto's full strength (the width mean-0511 was tuned at and v5s1 was gated at). Depth is
 * fixed per engine in the wasm (`Brain::new`: d10, `Brain::with_net`: d6). */
export const WIDTH = 96;

let replica: Game | null = null;
let brain: Brain | null = null;
/** The engine `brain` plays (the next round's too), and the one the page asked for. */
let active: Engine | null = null;
let wanted: Engine = 'hand';
let net: Net | null = null;
let netLoad: Promise<Net> | null = null;
/** The round's bot seed, so that a brain swapped in mid-round is seeded like one made at its start. */
let botSeed = 1;
const benchMs: Partial<Record<Engine, number>> = {};
const queue: ToWorker[] = [];
let booted = false;

function post(m: FromWorker): void {
  self.postMessage(m);
}

function makeBrain(engine: Engine, seed: number): Brain {
  return engine === 'nnue' && net ? Brain.with_net(net, WIDTH, seed) : new Brain(WIDTH, seed);
}

/** Time `n` moves of self-play from a fresh game: the device's think time per move, in ms (p90). */
function bench(engine: Engine, n: number): number {
  const g = new Game(0x5eed, 12, false);
  const b = makeBrain(engine, 1);
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

/** Switch Otto to `engine` (benchmarked the first time), mid-round too: the bot keeps no state that
 * matters between moves. */
function activate(engine: Engine): void {
  // The network is about 7× the work per move; fewer bench moves keep a mid-round switch's stall short.
  benchMs[engine] ??= bench(engine, engine === 'nnue' ? 16 : 24);
  if (active !== engine && replica) {
    brain?.free();
    brain = makeBrain(engine, botSeed);
  }
  active = engine;
  const name = engine === 'nnue' && net ? net.name() : 'mean-0511';
  post({ t: 'engine', engine, name, thinkMs: benchMs[engine] });
}

/** Fetch the network file, reporting progress; gunzip it unless the server already did. */
async function fetchNet(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (performance.now() - last > 100) {
      last = performance.now();
      post({ t: 'progress', loaded, total });
    }
  }
  post({ t: 'progress', loaded, total: Math.max(total, loaded) });
  let bytes = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) {
    bytes.set(c, at);
    at += c.length;
  }
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return bytes;
}

function loadNet(url: string, name: string): Promise<Net> {
  netLoad ??= fetchNet(url).then((bytes) => (net = new Net(bytes, name)));
  return netLoad;
}

function selectEngine(engine: Engine, netUrl: string, netName: string): void {
  wanted = engine;
  if (engine === 'hand' || net) {
    activate(engine);
    return;
  }
  // Until the network is here, Otto plays (or starts) with the hand-written engine.
  if (active === null) activate('hand');
  loadNet(netUrl, netName).then(
    () => {
      if (wanted === 'nnue') activate('nnue');
    },
    (e: unknown) => {
      netLoad = null; // allow a retry
      if (wanted === 'nnue') wanted = 'hand';
      post({ t: 'engineError', message: e instanceof Error ? e.message : String(e) });
    },
  );
}

function handle(m: ToWorker): void {
  switch (m.t) {
    case 'engine':
      selectEngine(m.engine, m.netUrl, m.netName);
      break;
    case 'reset':
      replica?.free();
      brain?.free();
      botSeed = m.seed ^ 0x9e3779b9;
      replica = new Game(m.seed, m.fpp, false);
      brain = makeBrain(active ?? 'hand', botSeed);
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
  booted = true;
  for (const m of queue.splice(0)) handle(m);
});
