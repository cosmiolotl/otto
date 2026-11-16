// Pack a value network for the web client: gzip a `.nnue` file into public/nets/<name>.nnue.gzip.
//
//   node scripts/pack-net.mjs <net.nnue> <name>
//
// The page fetches it only when the player picks the neural engine (src/settings.ts `NET`), and the worker
// gunzips it with `DecompressionStream`. gzip -9 takes a v5 net (L1 256) from 25.7 MB to about 14.1 MB.
// Not `.gz`: static servers (Vite's among them) send that with `Content-Encoding: gzip`, so the browser
// would decode it on the fly and the download progress would be measured against the wrong size.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { constants, gzipSync } from 'node:zlib';

const [src, name] = process.argv.slice(2);
if (!src || !name || !/^[\w.-]+$/.test(name)) {
  console.error('usage: node scripts/pack-net.mjs <net.nnue> <name>');
  process.exit(1);
}
const bytes = readFileSync(src);
if (bytes.subarray(0, 8).toString('latin1') !== 'OTTONNUE') {
  console.error(`${src}: not an otto network file`);
  process.exit(1);
}
const out = join(resolve(dirname(fileURLToPath(import.meta.url)), '..'), 'public', 'nets', `${name}.nnue.gzip`);
mkdirSync(dirname(out), { recursive: true });
const gz = gzipSync(bytes, { level: constants.Z_BEST_COMPRESSION });
writeFileSync(out, gz);
const mb = (n) => (n / 1e6).toFixed(1);
console.log(`${out}: ${mb(bytes.length)} MB → ${mb(gz.length)} MB`);
