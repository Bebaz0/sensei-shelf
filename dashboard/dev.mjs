// Runs the inspection API (port 8000) and the dashboard dev server together: `npm run dev:all`.
// Plain Node so it works the same on macOS, Linux and Windows. Ctrl+C stops both.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
const HERE = fileURLToPath(new URL('.', import.meta.url))
const win = process.platform === 'win32'

// Repo venv first (bin/ on macOS and Linux, Scripts\ on Windows), then whatever python is on PATH.
const venv = `${ROOT}.venv/${win ? 'Scripts/python.exe' : 'bin/python'}`
const python = existsSync(venv) ? venv : win ? 'python' : 'python3'

const procs = [
  ['api', python, ['-m', 'uvicorn', 'api.main:app', '--reload', '--port', '8000'], ROOT],
  ['web', process.execPath, ['node_modules/vite/bin/vite.js', ...process.argv.slice(2)], HERE],
].map(([name, cmd, args, cwd]) => {
  const p = spawn(cmd, args, { cwd, stdio: 'inherit' })
  p.on('error', (e) => {
    console.error(`${name}: could not start ${cmd} (${e.message})`)
    stop(1)
  })
  p.on('exit', (code) => {
    if (name === 'api' && code) console.error(`\nAPI exited (${code}). Missing packages? Run: ${python} -m pip install -r requirements.txt`)
    stop(code ?? 0)
  })
  return p
})

let stopping = false
function stop(code) {
  if (stopping) return
  stopping = true
  for (const p of procs) if (p.exitCode === null) p.kill()
  process.exitCode = code
}
process.on('SIGINT', () => stop(0))
process.on('SIGTERM', () => stop(0))
