// Build the engine (web/engine) to WebAssembly and generate the JS bindings into src/wasm/.
//
//   node scripts/build-wasm.mjs
//
// Needs the wasm32-unknown-unknown target and a wasm-bindgen CLI whose version matches the crate:
//   rustup target add wasm32-unknown-unknown
//   cargo install wasm-bindgen-cli --version <version in engine/Cargo.toml> --locked
import { execFileSync } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(web, '..');
const run = (cmd, args, opts = {}) => execFileSync(cmd, args, { stdio: 'inherit', cwd: root, ...opts });

const want = readFileSync(join(web, 'engine', 'Cargo.toml'), 'utf8').match(/wasm-bindgen = "=([^"]+)"/)[1];
let have = '';
try {
  have = execFileSync('wasm-bindgen', ['--version'], { encoding: 'utf8' }).trim().split(' ').pop();
} catch {
  /* not installed */
}
if (have !== want) {
  console.error(`wasm-bindgen CLI ${want} required (found ${have || 'none'}):`);
  console.error(`  cargo install wasm-bindgen-cli --version ${want} --locked`);
  process.exit(1);
}

run('cargo', ['build', '-p', 'otto-wasm', '--target', 'wasm32-unknown-unknown', '--profile', 'wasm']);
const wasm = join(root, 'target', 'wasm32-unknown-unknown', 'wasm', 'otto_wasm.wasm');
run('wasm-bindgen', [wasm, '--out-dir', join(web, 'src', 'wasm'), '--target', 'web', '--omit-default-module-path']);

const out = join(web, 'src', 'wasm', 'otto_wasm_bg.wasm');
console.log(`otto_wasm_bg.wasm: ${(statSync(out).size / 1024).toFixed(0)} KiB`);
