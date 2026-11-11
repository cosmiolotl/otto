import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs, so dist/ works from any path: a domain root or a subpath like
  // GitHub Pages' https://<user>.github.io/otto/.
  base: './',
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
