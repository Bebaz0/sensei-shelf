import { createReadStream, existsSync } from 'node:fs'
import { join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
const SYNCED = join(ROOT, 'dashboard/public/data')
const MOCK = join(ROOT, 'mocks/results.json')
const FRAMES = join(ROOT, 'data/Frutas e Legumes (train & test)')

// Dev only: until scripts/sync_dashboard.sh has filled public/data, serve the mock results and the raw frames.
function mockData(): Plugin {
  return {
    name: 'mock-data',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/data', (req, res, next) => {
        const path = decodeURIComponent(req.url!.split('?')[0])
        if (existsSync(join(SYNCED, path))) return next()
        const file = path === '/results.json' ? MOCK : path.startsWith('/frames/') ? resolve(FRAMES, path.slice('/frames/'.length)) : null
        if (!file || (file !== MOCK && !file.startsWith(FRAMES + sep)) || !existsSync(file)) return next()
        res.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : 'image/jpeg')
        createReadStream(file).pipe(res)
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), mockData()],
  // src/lib/results.ts imports data/planogram.json from the repo root
  server: { fs: { allow: [ROOT] } },
})
