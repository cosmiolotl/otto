// DOM for the two player clusters and the settings panel.
import { FPS, Info, Lock } from './engine';
import { Icons, icon, keycap } from './icons';
import { VIEW_ROWS, drawBoard, drawMeter, drawPiece } from './render';
import { ACTIONS, type Action, PPS_MAX, PPS_MIN, SDF_INFINITE, settings } from './settings';

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', text = ''): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function label(text: string, cls = ''): HTMLElement {
  return el('span', `mono-label ${cls}`.trim(), text);
}

const SPIN_NAMES = ['', 'MINI ', ''];
const LINE_NAMES = ['', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUAD'];
const PIECE_LETTERS = 'IOTLJSZ';

/** The clear type of a lock only: "T-SPIN MINI SINGLE", "QUAD". Empty for a plain non-clearing lock. */
export function describeLock(l: Int32Array): string {
  const lines = l[Lock.lines];
  const spin = l[Lock.spin];
  if (spin > 0) return `${PIECE_LETTERS[l[Lock.piece]]}-SPIN ${SPIN_NAMES[spin]}${LINE_NAMES[lines]}`.trim();
  return LINE_NAMES[lines];
}

function fmtTime(frames: number): string {
  const s = Math.floor(frames / FPS);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

export interface Snapshot {
  frame: number;
  board: Uint8Array;
  info: Int32Array;
  incoming: Int32Array;
  lock: Int32Array;
  showPiece: boolean;
}

export class PlayerView {
  readonly root = el('div', 'player');
  private board = el('canvas');
  private meter = el('canvas', 'meter');
  private hold = el('canvas');
  private next = Array.from({ length: 5 }, () => el('canvas'));
  private b2b = el('div', 'b2b', '0');
  private apm = el('span', 'value', '0.0');
  private pps = el('span', 'value', '0.00');
  private time = el('span', 'value', '00:00');
  private action = el('span', 'action');
  private winsEl = el('span');
  private overlay = el('div', 'overlay');
  private lastSeq = 0;
  private actionTimer = 0;

  constructor(name: string, meta: string, otto: boolean) {
    if (otto) this.root.classList.add('otto');
    const head = el('div', 'player-head');
    const wins = el('span', 'wins');
    wins.title = 'Rounds won this session';
    wins.append(icon(Icons.Trophy, 13), this.winsEl);
    this.winsEl.textContent = '0';
    head.append(el('span', 'name', name), el('span', 'meta', meta), this.action, wins);
    this.meta = head.children[1] as HTMLElement;

    const left = el('div', 'left-col');
    const holdBox = el('div', 'piece-box hold-box');
    holdBox.append(this.hold);
    const stats = el('div', 'stats');
    for (const [k, v] of [
      ['APM', this.apm],
      ['PPS', this.pps],
      ['TIME', this.time],
    ] as const) {
      const s = el('div', 'stat');
      s.append(label(k), v);
      stats.append(s);
    }
    left.append(label('Hold'), holdBox, label('B2B', 'b2b-label'), this.b2b, stats);

    const wrap = el('div', 'board-wrap');
    this.overlay.hidden = true;
    wrap.append(this.board, this.overlay);

    const nextCol = el('div', 'next-col');
    const nextBox = el('div', 'piece-box next-box');
    nextBox.append(...this.next);
    nextCol.append(label('Next'), nextBox);

    const row = el('div', 'player-row');
    row.append(left, this.meter, wrap, nextCol);
    this.root.append(head, row);
  }

  private meta: HTMLElement;

  setMeta(text: string): void {
    this.meta.textContent = text;
  }

  setWins(n: number): void {
    this.winsEl.textContent = String(n);
  }

  /** `null` hides the overlay. */
  setOverlay(o: { big: string; cls?: string; hint?: string; hintIcon?: SVGElement } | null): void {
    if (!o) {
      this.overlay.hidden = true;
      return;
    }
    this.overlay.hidden = false;
    this.overlay.replaceChildren();
    this.overlay.append(el('div', `big ${o.cls ?? ''}`.trim(), o.big));
    if (o.hint) {
      const h = el('div', 'hint');
      if (o.hintIcon) h.append(o.hintIcon);
      h.append(o.hint);
      this.overlay.append(h);
    }
  }

  resetRound(): void {
    this.lastSeq = 0;
    this.action.classList.remove('show');
    this.action.textContent = '';
  }

  draw(s: Snapshot, cell: number): void {
    drawBoard(this.board, cell, { cells: s.board, info: s.info, showPiece: s.showPiece });
    drawMeter(this.meter, cell, s.incoming, s.frame);
    const pc = Math.max(8, Math.round(cell * 0.43));
    drawPiece(this.hold, pc, s.info[Info.hold], s.info[Info.holdLocked] === 1);
    for (let i = 0; i < 5; i++) drawPiece(this.next[i], pc, s.info[Info.next + i]);

    const b2b = Math.max(0, s.info[Info.b2b]);
    this.b2b.textContent = String(b2b);
    const active = b2b >= 4;
    this.b2b.classList.toggle('active', active);
    if (active) {
      // Hue drifts from green through cyan and blue to violet as the chain grows; the spin speeds up a bit.
      this.b2b.style.setProperty('--hue', String(160 + Math.min(b2b - 4, 20) * 9));
      this.b2b.style.setProperty('--spin', `${Math.max(5, 16 - (b2b - 4) * 0.6)}s`);
    }

    const minutes = s.frame / FPS / 60;
    const pieces = s.info[Info.pieces];
    this.apm.textContent = minutes > 0 ? (s.info[Info.attack] / minutes).toFixed(1) : '0.0';
    this.pps.textContent = s.frame > 0 ? ((pieces * FPS) / s.frame).toFixed(2) : '0.00';
    this.time.textContent = fmtTime(s.frame);

    const seq = s.lock[Lock.seq];
    if (seq !== this.lastSeq) {
      this.lastSeq = seq;
      const text = describeLock(s.lock);
      if (text) {
        this.action.textContent = text;
        this.action.classList.add('show');
        window.clearTimeout(this.actionTimer);
        this.actionTimer = window.setTimeout(() => this.action.classList.remove('show'), 1400);
      }
    }
  }
}

/** Cell size that fits both boards and the settings panel in the window. */
export function fitCell(stage: HTMLElement): number {
  const h = window.innerHeight - 24 - 32 - 60 - 16 - 30;
  const byHeight = Math.floor(h / VIEW_ROWS);
  const w = stage.clientWidth;
  // Two clusters of (80 + 10 + 6 + 10 + 10·cell + 10 + 64) plus a 48 gap; one per row when wrapped.
  const twoUp = Math.floor((w - 48 - 2 * 180) / 20);
  const oneUp = Math.floor((w - 180) / 10);
  const byWidth = twoUp >= 16 ? twoUp : oneUp;
  return Math.max(14, Math.min(32, byHeight, byWidth));
}

function setFill(input: HTMLInputElement): void {
  const min = Number(input.min);
  const max = Number(input.max);
  input.style.setProperty('--fill', `${((Number(input.value) - min) / (max - min)) * 100}%`);
}

/** The settings panel: Otto's speed, handling, key bindings. */
export class SettingsPanel {
  private ppsValue = el('span', 'pps-value');
  private ppsInput = el('input');
  private deviceText = el('span');
  private warning = el('div', 'warning');
  private warningText = el('span');
  private footText = el('span');
  private footDot = el('span', 'dot');
  private handlingInputs: { key: keyof typeof HANDLING; input: HTMLInputElement; value: HTMLElement }[] = [];
  private controlRows = new Map<Action, HTMLButtonElement>();
  private listening: Action | null = null;
  /** Max PPS this device sustains at full search (null until measured). */
  deviceLimit: number | null = null;

  constructor(private root: HTMLElement) {
    this.build();
    settings.subscribe(() => this.sync());
    this.sync();
  }

  isListening(): boolean {
    return this.listening !== null;
  }

  /** A key press while rebinding: bind it (Escape cancels). Returns true if consumed. */
  capture(e: KeyboardEvent): boolean {
    if (!this.listening) return false;
    e.preventDefault();
    const a = this.listening;
    this.listening = null;
    if (e.code !== 'Escape') settings.bind(a, e.code);
    else this.sync();
    return true;
  }

  setDevice(limitPps: number | null, msPerMove: number | null): void {
    this.deviceLimit = limitPps;
    if (limitPps === null) {
      this.deviceText.replaceChildren('Measuring this device…');
    } else {
      const b = el('b', '', `${limitPps.toFixed(1)} pps`);
      this.deviceText.replaceChildren('This device keeps up to about ', b);
    }
    this.footDot.classList.toggle('live', msPerMove !== null);
    this.footText.textContent =
      msPerMove === null ? 'otto.wasm · loading' : `otto.wasm · local worker · ${msPerMove.toFixed(1)} ms/move`;
    this.syncWarning();
  }

  private build(): void {
    const head = el('div', 'settings-head');
    const logo = el('img');
    logo.src = `${import.meta.env.BASE_URL}icon.svg`;
    logo.alt = '';
    const reset = el('button', 'btn btn-ghost');
    reset.type = 'button';
    reset.append(icon(Icons.RotateCcw, 14), 'Reset');
    reset.title = 'Reset all settings to their defaults';
    reset.onclick = () => settings.reset();
    head.append(logo, el('span', 'title', 'Settings'), reset);

    // Speed.
    const speed = el('div', 'section');
    const sh = el('div', 'section-head');
    sh.append(label("Otto's speed"), this.ppsValue);
    const inp = this.ppsInput;
    inp.type = 'range';
    inp.min = String(PPS_MIN);
    inp.max = String(PPS_MAX);
    inp.step = '0.1';
    inp.setAttribute('aria-label', "Otto's speed in pieces per second");
    inp.oninput = () => settings.update((s) => (s.pps = Number(inp.value)));
    const ticks = el('div', 'ticks');
    for (const t of ['0.5', '2.5', '5', '7.5', '10']) ticks.append(el('span', '', t));
    const device = el('div', 'device');
    device.append(icon(Icons.Cpu, 14), this.deviceText);
    this.warning.append(icon(Icons.TriangleAlert, 15), this.warningText);
    this.warning.hidden = true;
    speed.append(sh, inp, ticks, device, this.warning);

    // Handling.
    const handling = el('div', 'section');
    handling.append(label('Handling'));
    for (const key of Object.keys(HANDLING) as (keyof typeof HANDLING)[]) {
      const spec = HANDLING[key];
      const row = el('div', 'handling-row');
      const input = el('input', 'quiet');
      input.type = 'range';
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
      input.setAttribute('aria-label', spec.title);
      const value = el('span', 'value');
      input.oninput = () => settings.update((s) => (s.handling[key] = Number(input.value)));
      const name = el('span', 'name', key.toUpperCase());
      name.title = spec.title;
      row.append(name, input, value);
      handling.append(row);
      this.handlingInputs.push({ key, input, value });
    }

    // Controls.
    const controls = el('div', 'section controls');
    controls.append(label('Controls'));
    for (const a of ACTIONS) {
      const row = el('div', 'control-row');
      const btn = el('button', 'kbd-btn');
      btn.type = 'button';
      btn.setAttribute('aria-label', `Rebind ${a.label}`);
      btn.onclick = () => {
        this.listening = a.id;
        this.sync();
        btn.blur();
      };
      row.append(el('span', 'label', a.label), btn);
      controls.append(row);
      this.controlRows.set(a.id, btn);
    }
    controls.append(el('div', 'hint-text', 'Click a key to rebind it. Esc cancels.'));

    const foot = el('div', 'settings-foot');
    foot.append(this.footDot, this.footText);

    this.root.append(head, speed, handling, controls, foot);
  }

  private sync(): void {
    const s = settings.value;
    this.ppsInput.value = String(s.pps);
    setFill(this.ppsInput);
    this.ppsValue.replaceChildren(s.pps.toFixed(1), el('span', 'unit', 'pps'));
    for (const h of this.handlingInputs) {
      const v = s.handling[h.key];
      h.input.value = String(v);
      setFill(h.input);
      h.value.textContent = HANDLING[h.key].format(v);
    }
    for (const [a, btn] of this.controlRows) {
      if (this.listening === a) {
        btn.classList.add('listening');
        btn.replaceChildren(el('span', 'listen', 'Press a key…'));
      } else {
        btn.classList.remove('listening');
        btn.replaceChildren(...s.keys[a].map(keycap));
        if (s.keys[a].length === 0) btn.append(el('span', 'kbd', '—'));
      }
    }
    this.syncWarning();
  }

  private syncWarning(): void {
    const limit = this.deviceLimit;
    const over = limit !== null && settings.value.pps > limit;
    this.warning.hidden = !over;
    this.ppsInput.style.setProperty('--accent', over ? 'var(--warning)' : 'var(--brand)');
    if (over) {
      this.warningText.textContent =
        `This device can only search fast enough for about ${limit.toFixed(1)} pps. ` +
        'Above that, Otto plays slower than the setting.';
    }
  }
}

const HANDLING = {
  das: { min: 0, max: 333, step: 1, title: 'Delayed auto shift', format: (v: number) => `${v} ms` },
  arr: { min: 0, max: 83, step: 1, title: 'Auto repeat rate', format: (v: number) => `${v} ms` },
  sdf: {
    min: 5,
    max: SDF_INFINITE,
    step: 1,
    title: 'Soft drop factor',
    format: (v: number) => (v >= SDF_INFINITE ? '∞' : `${v}×`),
  },
  dcd: { min: 0, max: 333, step: 1, title: 'DAS cut delay', format: (v: number) => `${v} ms` },
} as const;
