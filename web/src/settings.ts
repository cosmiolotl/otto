// Player settings: Otto's speed, handling and key bindings. Kept in localStorage (a per-browser
// convenience; everything still works when storage is unavailable).

export type Action =
  | 'moveLeft'
  | 'moveRight'
  | 'softDrop'
  | 'hardDrop'
  | 'rotateCw'
  | 'rotateCcw'
  | 'rotate180'
  | 'hold'
  | 'restart';

export const ACTIONS: { id: Action; label: string }[] = [
  { id: 'moveLeft', label: 'Move left' },
  { id: 'moveRight', label: 'Move right' },
  { id: 'softDrop', label: 'Soft drop' },
  { id: 'hardDrop', label: 'Hard drop' },
  { id: 'rotateCw', label: 'Rotate CW' },
  { id: 'rotateCcw', label: 'Rotate CCW' },
  { id: 'rotate180', label: 'Rotate 180' },
  { id: 'hold', label: 'Hold' },
  { id: 'restart', label: 'Restart' },
];

export interface Handling {
  /** Delayed auto shift, ms. */
  das: number;
  /** Auto repeat rate, ms (0 = instant). */
  arr: number;
  /** Soft drop factor; `SDF_INFINITE` = instant. */
  sdf: number;
  /** DAS cut delay, ms. */
  dcd: number;
}

export interface Settings {
  /** Otto's pieces per second. */
  pps: number;
  handling: Handling;
  /** `KeyboardEvent.code`s per action (first is shown first). */
  keys: Record<Action, string[]>;
}

export const PPS_MIN = 0.5;
export const PPS_MAX = 10;
export const SDF_INFINITE = 41;

export const DEFAULTS: Settings = {
  pps: 2.5,
  handling: { das: 117, arr: 0, sdf: SDF_INFINITE, dcd: 17 },
  keys: {
    moveLeft: ['ArrowLeft'],
    moveRight: ['ArrowRight'],
    softDrop: ['ArrowDown'],
    hardDrop: ['Space'],
    rotateCw: ['KeyX', 'ArrowUp'],
    rotateCcw: ['KeyZ'],
    rotate180: ['KeyA'],
    hold: ['KeyC', 'ShiftLeft'],
    restart: ['KeyR'],
  },
};

const STORAGE_KEY = 'otto.settings.v1';

function clone(s: Settings): Settings {
  return JSON.parse(JSON.stringify(s)) as Settings;
}

function load(): Settings {
  const s = clone(DEFAULTS);
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return s;
    const saved = JSON.parse(raw) as Partial<Settings>;
    if (typeof saved.pps === 'number') s.pps = clampPps(saved.pps);
    if (saved.handling) Object.assign(s.handling, saved.handling);
    if (saved.keys) {
      for (const a of ACTIONS) {
        const k = saved.keys[a.id];
        if (Array.isArray(k) && k.every((c) => typeof c === 'string')) s.keys[a.id] = k;
      }
    }
  } catch {
    /* storage blocked or corrupt: defaults */
  }
  return s;
}

export function clampPps(pps: number): number {
  return Math.min(PPS_MAX, Math.max(PPS_MIN, Math.round(pps * 10) / 10));
}

type Listener = (s: Settings) => void;

class Store {
  value: Settings = load();
  private listeners: Listener[] = [];

  subscribe(fn: Listener): void {
    this.listeners.push(fn);
  }

  update(fn: (s: Settings) => void): void {
    fn(this.value);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.value));
    } catch {
      /* not persisted; fine */
    }
    for (const l of this.listeners) l(this.value);
  }

  reset(): void {
    this.update((s) => Object.assign(s, clone(DEFAULTS)));
  }

  /** The action bound to `code`, if any. */
  actionFor(code: string): Action | undefined {
    return ACTIONS.find((a) => this.value.keys[a.id].includes(code))?.id;
  }

  /** Bind `code` to `action` alone, removing it from any other action. */
  bind(action: Action, code: string): void {
    this.update((s) => {
      for (const a of ACTIONS) s.keys[a.id] = s.keys[a.id].filter((c) => c !== code);
      s.keys[action] = [code];
    });
  }
}

export const settings = new Store();
