// Lucide icons (https://lucide.dev) as inline SVG elements.
import {
  ArrowBigUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  CornerDownLeft,
  Cpu,
  Delete,
  Flag,
  type IconNode,
  RefreshCw,
  RotateCcw,
  Space,
  TriangleAlert,
  Trophy,
  createElement,
} from 'lucide';

export function icon(node: IconNode, size = 16, strokeWidth = 1.75): SVGElement {
  return createElement(node, {
    width: size,
    height: size,
    'stroke-width': strokeWidth,
    'aria-hidden': 'true',
  });
}

export const Icons = { Cpu, Flag, RefreshCw, RotateCcw, TriangleAlert, Trophy };

/** Keys drawn as an icon instead of text. */
const KEY_ICONS: Record<string, IconNode> = {
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  Space,
  ShiftLeft: ArrowBigUp,
  ShiftRight: ArrowBigUp,
  Enter: CornerDownLeft,
  Backspace: Delete,
};

const KEY_NAMES: Record<string, string> = {
  ArrowLeft: 'Left arrow',
  ArrowRight: 'Right arrow',
  ArrowUp: 'Up arrow',
  ArrowDown: 'Down arrow',
  Space: 'Space',
  ShiftLeft: 'Left Shift',
  ShiftRight: 'Right Shift',
  ControlLeft: 'Ctrl',
  ControlRight: 'Ctrl',
  AltLeft: 'Alt',
  AltRight: 'Alt',
  Enter: 'Enter',
  Backspace: 'Backspace',
};

/** Short label of a `KeyboardEvent.code` (`KeyX` → `X`, `Digit1` → `1`, `Numpad4` → `Num 4`). */
export function keyLabel(code: string): string {
  if (KEY_NAMES[code]) return KEY_NAMES[code];
  if (code.startsWith('Key')) return code.slice(3);
  if (code.startsWith('Digit')) return code.slice(5);
  if (code.startsWith('Numpad')) return `Num ${code.slice(6)}`;
  return code.replace(/(Left|Right)$/, '');
}

/** A keycap for `code`: an icon where one fits, else the key's label. */
export function keycap(code: string): HTMLElement {
  const k = document.createElement('kbd');
  k.className = 'kbd';
  k.title = keyLabel(code);
  const node = KEY_ICONS[code];
  if (node) {
    k.append(icon(node, 13, 2));
    if (code === 'Space') k.classList.add('kbd-wide');
  } else k.textContent = keyLabel(code);
  return k;
}
