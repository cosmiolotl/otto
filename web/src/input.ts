// The human's piece in real time: handling (DAS / ARR / SDF / DCD), gravity and lock delay, on top of the
// sim's `press()` keys. Timings follow TETR.IO S2 (docs/RULES.md §8): gravity 0.02 G growing by 0.0035/s
// after 2:00, a piece locks after 31 grounded frames or once it has used 15 lock resets.
import { FPS, Game, Key, Press } from './engine';
import { type Action, type Handling, SDF_INFINITE } from './settings';

const LOCK_FRAMES = 30;
const LOCK_RESETS = 15;
const GRAVITY = 0.02;
const GRAVITY_INCREASE = 0.0035;
const GRAVITY_MARGIN = 7200;

const msToFrames = (ms: number) => (ms * FPS) / 1000;

export class HumanController {
  private held = new Set<Action>();
  /** Direction of the active shift key (-1, 0, 1) and how long it has been held, in frames. */
  private dasDir = 0;
  private dasCharge = 0;
  private arrAcc = 0;
  private gravAcc = 0;
  private lockTimer = 0;
  private lockResets = 0;
  private lowestY = 99;
  /** Called after every lock of the human's piece. */
  onLock: () => void = () => {};

  constructor(
    private game: Game,
    private handling: () => Handling,
  ) {
    this.newPiece();
  }

  setGame(game: Game): void {
    this.game = game;
    this.newPiece();
  }

  /** Forget held keys (focus lost). */
  releaseAll(): void {
    this.held.clear();
    this.dasDir = 0;
  }

  /** Key went down (no auto-repeat events). `active`: the round is running (else only track DAS). */
  down(a: Action, active: boolean): void {
    if (this.held.has(a)) return;
    this.held.add(a);
    if (a === 'moveLeft' || a === 'moveRight') {
      this.dasDir = a === 'moveLeft' ? -1 : 1;
      this.dasCharge = 0;
      this.arrAcc = 0;
      if (active) this.shift(this.dasDir);
      return;
    }
    if (!active) return;
    switch (a) {
      case 'rotateCw':
        this.rotate(Key.RotateCw);
        break;
      case 'rotateCcw':
        this.rotate(Key.RotateCcw);
        break;
      case 'rotate180':
        this.rotate(Key.Rotate180);
        break;
      case 'hold':
        if (this.game.press(Key.Hold) === Press.Moved) this.spawned();
        break;
      case 'hardDrop':
        if (this.game.press(Key.HardDrop) === Press.Locked) this.locked();
        break;
      case 'softDrop':
        this.softDropFrame();
        break;
    }
  }

  up(a: Action): void {
    this.held.delete(a);
    if (a === 'moveLeft' || a === 'moveRight') {
      const other = a === 'moveLeft' ? 'moveRight' : 'moveLeft';
      if (this.held.has(other)) {
        this.dasDir = other === 'moveLeft' ? -1 : 1;
        this.dasCharge = 0;
        this.arrAcc = 0;
      } else this.dasDir = 0;
    }
  }

  /** One frame of the game clock (call after `game.advance_to(frame)`). */
  frame(frame: number): void {
    const h = this.handling();
    // Auto shift.
    if (this.dasDir !== 0) {
      this.dasCharge++;
      const das = msToFrames(h.das);
      if (this.dasCharge >= das) {
        const arr = msToFrames(h.arr);
        if (arr < 1) {
          if (this.game.press(this.dasDir < 0 ? Key.DasLeft : Key.DasRight) === Press.Moved) this.moved();
        } else {
          this.arrAcc++;
          while (this.arrAcc >= arr) {
            this.arrAcc -= arr;
            this.shift(this.dasDir);
          }
        }
      }
    }
    // Gravity and soft drop.
    const g = GRAVITY + (frame > GRAVITY_MARGIN ? ((frame - GRAVITY_MARGIN) * GRAVITY_INCREASE) / FPS : 0);
    if (this.held.has('softDrop')) {
      if (h.sdf >= SDF_INFINITE) this.softDropFrame();
      else this.gravAcc += Math.max(g * h.sdf, 0.05 * h.sdf);
    } else this.gravAcc += g;
    while (this.gravAcc >= 1) {
      if (this.game.press(Key.SoftDropStep) !== Press.Moved) {
        this.gravAcc = 0;
        break;
      }
      this.gravAcc -= 1;
      this.fell();
    }
    // Lock delay.
    if (this.game.grounded()) {
      this.gravAcc = 0;
      this.lockTimer++;
      if (this.lockTimer > LOCK_FRAMES || this.lockResets >= LOCK_RESETS) {
        this.game.gravity_lock();
        this.locked();
      }
    }
  }

  private softDropFrame(): void {
    if (this.handling().sdf >= SDF_INFINITE && this.game.press(Key.SoftDrop) === Press.Moved) this.fell();
  }

  private shift(dir: number): void {
    if (this.game.press(dir < 0 ? Key.MoveLeft : Key.MoveRight) === Press.Moved) this.moved();
  }

  private rotate(key: number): void {
    if (this.game.press(key) === Press.Moved) {
      this.moved();
      this.cutDas();
    }
  }

  /** A successful shift or rotation: spends a lock reset and restarts the lock timer. */
  private moved(): void {
    this.lockResets = Math.min(this.lockResets + 1, 31);
    if (this.lockResets < LOCK_RESETS) this.lockTimer = 0;
  }

  private fell(): void {
    const y = this.game.falling_y();
    if (y < this.lowestY) {
      this.lowestY = y;
      this.lockResets = 0;
      this.lockTimer = 0;
    }
  }

  /** DCD: after a rotation or a new piece, a charged auto shift waits `dcd` before moving again. */
  private cutDas(): void {
    const h = this.handling();
    const das = msToFrames(h.das);
    const dcd = msToFrames(h.dcd);
    if (dcd > 0 && this.dasCharge >= das) this.dasCharge = Math.max(0, das - dcd);
  }

  private locked(): void {
    this.spawned();
    this.onLock();
  }

  private spawned(): void {
    this.newPiece();
    this.cutDas();
  }

  private newPiece(): void {
    this.gravAcc = 0;
    this.lockTimer = 0;
    this.lockResets = 0;
    this.lowestY = this.game.falling_y();
  }
}
