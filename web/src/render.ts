// Canvas drawing: boards, piece previews and the garbage meter. Blocks are square tiles with connected
// textures: cells of the same placed piece merge into one shape, outlined only on its outer edges.
import { Info, RENDER_ROWS } from './engine';

/** Piece colours by sim piece index (I, O, T, L, J, S, Z). */
export const PIECE_COLORS = ['#2fb7c9', '#e8b931', '#a06cd5', '#e8833a', '#4c6fe0', '#34d399', '#e5484d'];
const GARBAGE = 8;
const GARBAGE_FILL = '#3a3a3a';
const BOARD_BG = '#141414';
const GRID = '#1f1f1f';
const LOCKED_HOLD = '#5a5a5a';

/** Screen-space link bits: the neighbour on that side is part of the same shape. The engine stores the
 * same bits (up, right, down, left) in the high nibble of each board cell. */
const N = 1;
const E = 2;
const S = 4;
const W = 8;

/** Spawn-orientation cells (x right, y down) per piece index. */
const SHAPES: [number, number][][] = [
  [[0, 0], [1, 0], [2, 0], [3, 0]],
  [[0, 0], [1, 0], [0, 1], [1, 1]],
  [[1, 0], [0, 1], [1, 1], [2, 1]],
  [[2, 0], [0, 1], [1, 1], [2, 1]],
  [[0, 0], [0, 1], [1, 1], [2, 1]],
  [[1, 0], [2, 0], [0, 1], [1, 1]],
  [[0, 0], [1, 0], [1, 1], [2, 1]],
];

/** Visible matrix rows plus the two rows above it where pieces spawn. */
export const VIEW_ROWS = 22;
const MATRIX_ROWS = 20;

export function sizeCanvas(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
  }
  const ctx = c.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

interface Tone {
  fill: string;
  edge: string;
  fillAlpha?: number;
  edgeAlpha?: number;
}

const tones = new Map<string, Tone>();

/** Fill plus a darker edge for a solid block of colour `c`. */
function tone(c: string): Tone {
  let t = tones.get(c);
  if (!t) {
    const n = parseInt(c.slice(1), 16);
    const k = (v: number) => Math.round(v * 0.62);
    const edge = `rgb(${k(n >> 16)}, ${k((n >> 8) & 255)}, ${k(n & 255)})`;
    t = { fill: c, edge };
    tones.set(c, t);
  }
  return t;
}

function cellTone(code: number): Tone {
  return tone(code === GARBAGE ? GARBAGE_FILL : code < 0 ? LOCKED_HOLD : PIECE_COLORS[code - 1]);
}

/**
 * One tile at grid cell (`gx`, `gy`) (screen space, y down) of size `s`, offset by (`ox`, `oy`). `linkAt`
 * gives the link bits of any grid cell (0 if empty), so outer edges and inner corners get an edge.
 */
function tile(
  ctx: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  s: number,
  gx: number,
  gy: number,
  linkAt: (x: number, y: number) => number,
  t: Tone,
): void {
  const l = linkAt(gx, gy);
  const x0 = Math.round(ox + gx * s);
  const y0 = Math.round(oy + gy * s);
  const x1 = Math.round(ox + (gx + 1) * s);
  const y1 = Math.round(oy + (gy + 1) * s);
  const w = x1 - x0;
  const h = y1 - y0;
  const b = Math.max(1, Math.round(s * 0.12));

  ctx.globalAlpha = t.fillAlpha ?? 1;
  ctx.fillStyle = t.fill;
  ctx.fillRect(x0, y0, w, h);

  ctx.globalAlpha = t.edgeAlpha ?? 1;
  ctx.fillStyle = t.edge;
  // Outer edges (corners shared by two edges are drawn once so translucent tones stay even).
  if (!(l & N)) ctx.fillRect(x0, y0, w, b);
  if (!(l & S)) ctx.fillRect(x0, y1 - b, w, b);
  const top = l & N ? y0 : y0 + b;
  const bot = l & S ? y1 : y1 - b;
  if (!(l & W)) ctx.fillRect(x0, top, b, bot - top);
  if (!(l & E)) ctx.fillRect(x1 - b, top, b, bot - top);
  // Inner corners: both sides linked but the diagonal cell is not part of the shape.
  if (l & N && l & E && !(linkAt(gx, gy - 1) & E)) ctx.fillRect(x1 - b, y0, b, b);
  if (l & S && l & E && !(linkAt(gx, gy + 1) & E)) ctx.fillRect(x1 - b, y1 - b, b, b);
  if (l & S && l & W && !(linkAt(gx, gy + 1) & W)) ctx.fillRect(x0, y1 - b, b, b);
  if (l & N && l & W && !(linkAt(gx, gy - 1) & W)) ctx.fillRect(x0, y0, b, b);
  ctx.globalAlpha = 1;
}

/** Draw the cells of one piece (screen grid coords) as a single connected shape. */
function shape(
  ctx: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  s: number,
  cells: [number, number][],
  t: Tone,
): void {
  const has = (x: number, y: number) => cells.some(([cx, cy]) => cx === x && cy === y);
  const linkAt = (x: number, y: number) =>
    has(x, y)
      ? (has(x, y - 1) ? N : 0) | (has(x + 1, y) ? E : 0) | (has(x, y + 1) ? S : 0) | (has(x - 1, y) ? W : 0)
      : 0;
  for (const [x, y] of cells) tile(ctx, ox, oy, s, x, y, linkAt, t);
}

export interface BoardView {
  cells: Uint8Array;
  info: Int32Array;
  /** Draw the falling piece and its ghost. */
  showPiece: boolean;
}

/** Draw a board: `VIEW_ROWS` rows, the top two (spawn area) without background. */
export function drawBoard(c: HTMLCanvasElement, cell: number, v: BoardView): void {
  const w = 10 * cell;
  const h = VIEW_ROWS * cell;
  const ctx = sizeCanvas(c, w, h);
  ctx.clearRect(0, 0, w, h);
  const top = (VIEW_ROWS - MATRIX_ROWS) * cell;
  ctx.fillStyle = BOARD_BG;
  ctx.fillRect(0, top, w, MATRIX_ROWS * cell);
  ctx.fillStyle = GRID;
  for (let x = 1; x < 10; x++) ctx.fillRect(x * cell, top, 1, MATRIX_ROWS * cell);
  for (let y = 1; y < MATRIX_ROWS; y++) ctx.fillRect(0, top + y * cell, w, 1);

  // Screen row `y` is board row `VIEW_ROWS - 1 - y`.
  const kind = (x: number, y: number) => {
    const row = VIEW_ROWS - 1 - y;
    return x < 0 || x >= 10 || row < 0 || row >= RENDER_ROWS ? 0 : v.cells[row * 10 + x] & 15;
  };
  const linkAt = (x: number, y: number) => {
    const k = kind(x, y);
    if (k !== GARBAGE) return k ? v.cells[(VIEW_ROWS - 1 - y) * 10 + x] >> 4 : 0;
    // Garbage merges with all neighbouring garbage.
    return (
      (kind(x, y - 1) === GARBAGE ? N : 0) |
      (kind(x + 1, y) === GARBAGE ? E : 0) |
      (kind(x, y + 1) === GARBAGE ? S : 0) |
      (kind(x - 1, y) === GARBAGE ? W : 0)
    );
  };
  for (let y = Math.max(0, VIEW_ROWS - RENDER_ROWS); y < VIEW_ROWS; y++) {
    for (let x = 0; x < 10; x++) {
      const k = kind(x, y);
      if (k) tile(ctx, 0, 0, cell, x, y, linkAt, cellTone(k));
    }
  }
  if (!v.showPiece) return;
  const color = v.info[Info.piece] + 1;
  const pc = PIECE_COLORS[color - 1];
  const cellsAt = (base: number) => {
    const out: [number, number][] = [];
    for (let i = 0; i < 4; i++) {
      const y = v.info[base + 2 * i + 1];
      if (y < VIEW_ROWS) out.push([v.info[base + 2 * i], VIEW_ROWS - 1 - y]);
    }
    return out;
  };
  // Ghost: translucent fill with the outline in the piece colour.
  shape(ctx, 0, 0, cell, cellsAt(Info.ghost), { fill: pc, edge: pc, fillAlpha: 0.14, edgeAlpha: 0.6 });
  shape(ctx, 0, 0, cell, cellsAt(Info.cells), cellTone(color));
}

/** A piece centred in a `4 × 2` cell box. `piece` < 0 draws nothing; `dim` greys it out (hold locked). */
export function drawPiece(c: HTMLCanvasElement, cell: number, piece: number, dim = false): void {
  const w = 4 * cell;
  const h = 2 * cell;
  const ctx = sizeCanvas(c, w, h);
  ctx.clearRect(0, 0, w, h);
  if (piece < 0) return;
  const cells = SHAPES[piece];
  const cw = Math.max(...cells.map(([x]) => x)) + 1;
  const ch = Math.max(...cells.map(([, y]) => y)) + 1;
  shape(ctx, (w - cw * cell) / 2, (h - ch * cell) / 2, cell, cells, cellTone(dim ? -1 : piece + 1));
}

/** Incoming garbage, stacked from the bottom: red once it can enter the board, amber while in flight. */
export function drawMeter(c: HTMLCanvasElement, cell: number, incoming: Int32Array, frame: number): void {
  const w = 6;
  const h = MATRIX_ROWS * cell;
  const ctx = sizeCanvas(c, w, h);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1c1c1c';
  ctx.beginPath();
  ctx.roundRect(0, 0, w, h, 3);
  ctx.fill();
  let y = h;
  for (let i = 0; i < incoming.length && y > 0; i += 2) {
    const lines = incoming[i];
    const ready = frame >= incoming[i + 1];
    const seg = Math.min(y, lines * cell);
    ctx.fillStyle = ready ? '#e5484d' : '#ffb224';
    ctx.beginPath();
    ctx.roundRect(0, y - seg, w, seg - 2, 3);
    ctx.fill();
    y -= seg;
  }
}
