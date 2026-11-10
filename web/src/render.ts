// Canvas drawing: boards, piece previews and the garbage meter, in the "outline" block style of the
// Studio design (tinted fill + 1px stroke in the piece colour).
import { Info, RENDER_ROWS } from './engine';

/** Piece colours by sim piece index (I, O, T, L, J, S, Z). */
export const PIECE_COLORS = ['#2fb7c9', '#e8b931', '#a06cd5', '#e8833a', '#4c6fe0', '#34d399', '#e5484d'];
const GARBAGE_FILL = '#262626';
const GARBAGE_STROKE = '#3a3a3a';
const BOARD_BG = '#141414';
const GRID = '#1f1f1f';
const LOCKED_HOLD = '#5a5a5a';

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

function block(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, color: number, alpha = 1): void {
  const inset = Math.max(1, Math.round(s * 0.06));
  const r = Math.max(1, s * 0.08);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.roundRect(x + inset + 0.5, y + inset + 0.5, s - 2 * inset - 1, s - 2 * inset - 1, r);
  if (color === 8) {
    ctx.fillStyle = GARBAGE_FILL;
    ctx.fill();
    ctx.strokeStyle = GARBAGE_STROKE;
  } else {
    const c = color < 0 ? LOCKED_HOLD : PIECE_COLORS[color - 1];
    ctx.fillStyle = c;
    ctx.globalAlpha = alpha * 0.24;
    ctx.fill();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = c;
  }
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.globalAlpha = 1;
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

  const py = (row: number) => (VIEW_ROWS - 1 - row) * cell;
  for (let row = 0; row < Math.min(VIEW_ROWS, RENDER_ROWS); row++) {
    for (let x = 0; x < 10; x++) {
      const k = v.cells[row * 10 + x];
      if (k) block(ctx, x * cell, py(row), cell, k);
    }
  }
  if (!v.showPiece) return;
  const color = v.info[Info.piece] + 1;
  // Ghost: dashed outline.
  const pc = PIECE_COLORS[color - 1];
  ctx.setLineDash([Math.max(2, cell * 0.16), Math.max(2, cell * 0.1)]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = pc;
  for (let i = 0; i < 4; i++) {
    const gx = v.info[Info.ghost + 2 * i];
    const gy = v.info[Info.ghost + 2 * i + 1];
    if (gy >= VIEW_ROWS) continue;
    const inset = Math.max(2, cell * 0.1);
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = pc;
    ctx.fillRect(gx * cell + inset, py(gy) + inset, cell - 2 * inset, cell - 2 * inset);
    ctx.globalAlpha = 0.9;
    ctx.strokeRect(gx * cell + inset, py(gy) + inset, cell - 2 * inset, cell - 2 * inset);
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  for (let i = 0; i < 4; i++) {
    const x = v.info[Info.cells + 2 * i];
    const y = v.info[Info.cells + 2 * i + 1];
    if (y < VIEW_ROWS) block(ctx, x * cell, py(y), cell, color);
  }
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
  const ox = (w - cw * cell) / 2;
  const oy = (h - ch * cell) / 2;
  for (const [x, y] of cells) block(ctx, ox + x * cell, oy + y * cell, cell, dim ? -1 : piece + 1);
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
