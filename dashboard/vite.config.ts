import { fileURLToPath } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const ROOT = fileURLToPath(new URL('..', import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // src/lib/results.ts imports data/planogram.json from the repo root
  server: {
    fs: { allow: [ROOT] },
    proxy: { '/data': { target: 'http://localhost:8000', changeOrigin: true } },
  },
})
