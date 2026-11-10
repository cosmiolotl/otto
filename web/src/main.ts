// The match: the authoritative `Game` on the page, a 60 Hz frame clock, the human controller, and Otto's
// moves from the worker (which keeps a replica of the game by replaying our event log).
import './style.css';
import { FPS, Game, loadEngine } from './engine';
import { Icons, icon } from './icons';
import { HumanController } from './input';
import type { FromWorker, ToWorker } from './protocol';
import { settings } from './settings';
import { PlayerView, SettingsPanel, fitCell } from './ui';

type Phase = 'loading' | 'idle' | 'countdown' | 'playing' | 'over';

const $ = (id: string) => document.getElementById(id)!;
const stage = $('stage');
const you = new PlayerView('You', 'keyboard', false);
const otto = new PlayerView('Otto', '', true);
stage.append(you.root, otto.root);
const panel = new SettingsPanel($('settings'));
$('restart').append(icon(Icons.RefreshCw, 14), 'Restart');
$('forfeit').append(icon(Icons.Flag, 14), 'Forfeit');

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToWorker, transfer: Transferable[] = []) => worker.postMessage(m, transfer);

let phase: Phase = 'loading';
let game: Game | null = null;
let human: HumanController | null = null;
let round = 0;
let seed = 0;
/** `performance.now()` of frame 0 of the round (moved forward while the tab is hidden). */
let t0 = 0;
let hiddenAt = 0;
let countdownStart = 0;
const wins = [0, 0];

// Otto's move pipeline: request a decision a few frames before it is due, play it when due.
let moveId = 0;
let requested = false;
let decision: { id: number; packed: number } | null = null;
/** Worker think time per move: benchmark p90, then a running average from play. */
let benchMs: number | null = null;
let thinkEma: number | null = null;

const ottoFpp = () => Math.max(1, Math.round(FPS / settings.value.pps));

function deviceLimit(): number | null {
  if (benchMs === null) return null;
  const ms = Math.max(benchMs, thinkEma ?? 0);
  // One move per piece, plus messaging and frame-alignment slack.
  return Math.floor((1000 / (ms * 1.25 + 4)) * 10) / 10;
}

function updateDevice(): void {
  panel.setDevice(deviceLimit(), thinkEma ?? benchMs);
  otto.setMeta(`${settings.value.pps.toFixed(1)} pps`);
}

settings.subscribe(() => {
  game?.set_otto_fpp(ottoFpp());
  updateDevice();
});

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  if (m.t === 'ready') {
    benchMs = m.thinkMs;
    updateDevice();
    phase = 'idle';
    showIdle();
  } else if (m.t === 'decision') {
    if (m.round !== round || m.id !== moveId) return;
    thinkEma = thinkEma === null ? m.ms : thinkEma * 0.9 + m.ms * 0.1;
    if (moveId % 10 === 0) updateDevice();
    if (m.packed < 0) {
      // No legal placement: Otto is topped out.
      endRound(0);
      return;
    }
    decision = { id: m.id, packed: m.packed };
  }
};

function flushLog(): void {
  if (!game) return;
  const events = game.take_log();
  if (events.length) send({ t: 'log', events }, [events.buffer]);
}

function humanFpp(): number {
  if (!game) return 30;
  const info = game.info(0);
  const pieces = info[27];
  return pieces > 0 ? Math.min(120, Math.max(6, Math.round(game.frame() / pieces))) : 30;
}

/** Frames before Otto's lock to ask for its decision: covers the think time plus a little slack. */
function lead(): number {
  const ms = thinkEma ?? benchMs ?? 10;
  return Math.min(ottoFpp(), Math.ceil((ms * FPS) / 1000) + 2);
}

function ottoFrame(f: number): void {
  if (!game) return;
  const due = game.otto_due();
  if (decision && f >= due) {
    const ok = game.play_otto(decision.packed);
    decision = null;
    requested = false;
    moveId++;
    if (!ok) console.warn('otto: placement rejected; asking again');
    flushLog();
  }
  if (!requested && f >= due - lead()) {
    flushLog();
    send({ t: 'think', id: moveId, round, frame: f, humanFpp: humanFpp() });
    requested = true;
  }
}

/** Run the clock up to `now`. */
function stepTo(now: number): void {
  if (phase !== 'playing' || !game || !human) return;
  const target = Math.floor(((now - t0) * FPS) / 1000);
  // A long stall (debugger, sleep) should not replay seconds of gravity in one go.
  if (target - game.frame() > FPS) t0 = now - (game.frame() * 1000) / FPS;
  const to = Math.min(target, game.frame() + FPS);
  while (game.frame() < to && phase === 'playing') {
    const f = game.frame() + 1;
    game.advance_to(f);
    human.frame(f);
    if (checkWinner()) return;
    ottoFrame(f);
    if (checkWinner()) return;
  }
  flushLog();
}

function checkWinner(): boolean {
  const w = game?.winner() ?? -1;
  if (w >= 0) endRound(w);
  return w >= 0;
}

function startRound(): void {
  if (phase === 'loading') return;
  round++;
  seed = (Math.random() * 0x7fffffff) >>> 0;
  game?.free();
  game = new Game(seed, ottoFpp(), true);
  if (human) human.setGame(game);
  else {
    human = new HumanController(game, () => settings.value.handling);
    human.onLock = () => checkWinner();
  }
  send({ t: 'reset', seed, fpp: ottoFpp() });
  moveId = 0;
  requested = false;
  decision = null;
  you.resetRound();
  otto.resetRound();
  $('seed').textContent = `0x${seed.toString(16).padStart(8, '0')}`;
  phase = 'countdown';
  countdownStart = performance.now();
  otto.setOverlay(null);
}

function endRound(winner: number): void {
  if (phase === 'over') return;
  phase = 'over';
  wins[winner]++;
  you.setWins(wins[0]);
  otto.setWins(wins[1]);
  $('wins-0').textContent = String(wins[0]);
  $('wins-1').textContent = String(wins[1]);
  const hint = { hint: 'Enter for the next round', hintIcon: icon(Icons.RefreshCw, 13) };
  you.setOverlay(winner === 0 ? { big: 'You win', cls: 'win', ...hint } : { big: 'Topped out', cls: 'loss', ...hint });
  otto.setOverlay(winner === 1 ? { big: 'Otto wins', cls: 'win' } : { big: 'Topped out', cls: 'loss' });
}

function showIdle(): void {
  you.setOverlay({ big: 'Ready?', hint: 'Press Enter to start' });
  otto.setOverlay(null);
}

function forfeit(): void {
  if (phase === 'playing' || phase === 'countdown') endRound(1);
}

$('restart').onclick = () => startRound();
$('forfeit').onclick = () => forfeit();

// ---- keyboard ----

window.addEventListener('keydown', (e) => {
  if (panel.capture(e)) return;
  const target = e.target as HTMLElement;
  const inControl = target instanceof HTMLInputElement || target instanceof HTMLButtonElement;
  if (e.code === 'Enter' && (phase === 'idle' || phase === 'over') && !(target instanceof HTMLButtonElement)) {
    e.preventDefault();
    startRound();
    return;
  }
  const action = settings.actionFor(e.code);
  if (!action) return;
  // While a round runs every bound key belongs to the game (even with a slider focused).
  if (inControl && phase !== 'playing' && phase !== 'countdown') return;
  e.preventDefault();
  if (inControl) target.blur();
  if (e.repeat) return;
  if (action === 'restart') {
    startRound();
    return;
  }
  if (!human) return;
  stepTo(performance.now());
  human.down(action, phase === 'playing');
  if (phase === 'playing') {
    checkWinner();
    flushLog();
  }
});

window.addEventListener('keyup', (e) => {
  const action = settings.actionFor(e.code);
  if (action && human) human.up(action);
});

window.addEventListener('blur', () => human?.releaseAll());

document.addEventListener('visibilitychange', () => {
  const now = performance.now();
  if (document.hidden) hiddenAt = now;
  else if (hiddenAt) {
    // Pause the clock while the tab is hidden (frames are wall-clock driven).
    if (phase === 'playing') t0 += now - hiddenAt;
    if (phase === 'countdown') countdownStart += now - hiddenAt;
    hiddenAt = 0;
  }
});

// ---- frame loop ----

const COUNT_MS = 600;
let cell = 0;

function resize(): void {
  cell = fitCell(stage);
  document.documentElement.style.setProperty('--cell', `${cell}px`);
}
window.addEventListener('resize', resize);
resize();

function render(): void {
  if (!game) return;
  const f = game.frame();
  for (const [p, view] of [you, otto].entries()) {
    view.draw(
      {
        frame: f,
        board: game.board(p),
        info: game.info(p),
        incoming: game.incoming(p),
        lock: game.last_lock(p),
        showPiece: phase === 'playing' || phase === 'countdown',
      },
      cell,
    );
  }
}

function loop(now: number): void {
  if (phase === 'countdown') {
    const left = 3 - Math.floor((now - countdownStart) / COUNT_MS);
    if (left <= 0) {
      phase = 'playing';
      t0 = now;
      you.setOverlay(null);
    } else you.setOverlay({ big: String(left), cls: 'count' });
  }
  stepTo(now);
  render();
  requestAnimationFrame(loop);
}

panel.setDevice(null, null);
you.setOverlay({ big: 'Loading Otto…' });
otto.setMeta(`${settings.value.pps.toFixed(1)} pps`);
loadEngine().then(() => {
  // Draw empty boards behind the overlay until the first round starts.
  game = new Game(1, ottoFpp(), false);
  render();
  game.free();
  game = null;
  requestAnimationFrame(loop);
});
