import { useEffect, useState } from 'react'
import { loadResults, type Results } from './lib/results'
import { type Cell, cellsOf, count, type Filter } from './lib/status'
import AreaDetail from './sections/AreaDetail'
import CameraList from './sections/CameraList'
import CameraView from './sections/CameraView'
import Header from './sections/Header'
import Kpis from './sections/Kpis'
import TaskPanel, { type Tab } from './sections/TaskPanel'
import Timeline from './sections/Timeline'

type Flags = Record<string, boolean>

// Adds the key when missing, removes it when present.
const toggle = (flags: Flags, key: string): Flags => {
  const next = { ...flags }
  if (next[key]) delete next[key]
  else next[key] = true
  return next
}

function App() {
  const [data, setData] = useState<Results | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [camId, setCamId] = useState<string | null>(null) // null = most urgent camera
  const [frame, setFrame] = useState<number | null>(null) // capture index, null = latest
  const [sel, setSel] = useState<string | null>(null) // selected area_id
  const [filter, setFilter] = useState<Filter>('all')
  const [tab, setTab] = useState<Tab>('tasks')
  const [resolved, setResolved] = useState<Flags>({}) // resolved this shift, local only for now
  const [flags, setFlags] = useState<Flags>({}) // reported as wrong detection, local only for now

  useEffect(() => {
    loadResults().then(setData, (e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="grid min-h-svh place-items-center px-4 text-center text-muted">{error}</div>
  if (!data) return <div className="grid min-h-svh place-items-center text-muted">Loading results…</div>

  // Latest capture of every camera, most urgent camera first
  const cams = Object.entries(data)
    .map(([id, camera]) => ({ id, cells: cellsOf(camera, camera.captures.at(-1)!, resolved), thumb: camera.captures.at(-1)!.image }))
    .sort((a, b) => score(b.cells) - score(a.cells))
  const allCells = cams.flatMap((c) => c.cells.map((cell) => ({ ...cell, cam: c.id })))
  const updated = Math.max(...Object.values(data).map((camera) => camera.captures.at(-1)!.ts))

  // Selected camera and capture
  const cam = camId ?? cams[0].id
  const camera = data[cam]
  const last = camera.captures.length - 1
  const fi = frame ?? last
  const capture = camera.captures[fi]
  const cells = cellsOf(camera, capture, fi === last ? resolved : {})
  const selected = cells.find((c) => c.area.area_id === sel)

  const pickCam = (id: string, areaId: string | null = null) => {
    setCamId(id)
    setFrame(null)
    setSel(areaId)
  }

  return (
    <div className="mx-auto flex max-w-[1480px] flex-col gap-[22px] px-4 pt-6 pb-10 sm:px-7">
      <Header updated={updated} />
      <Kpis cells={allCells} filter={filter} onFilter={setFilter} />

      <div className="grid items-start gap-[22px] lg:grid-cols-[250px_minmax(0,1fr)] xl:grid-cols-[262px_minmax(0,1fr)_340px]">
        <CameraList cams={cams} active={cam} onPick={pickCam} />

        <section aria-label="Selected camera" className="flex min-w-0 flex-col gap-4">
          <CameraView cam={cam} capture={capture} cells={cells} sel={sel} filter={filter} onSelect={setSel} />
          <Timeline camera={camera} frame={fi} done={resolved} onPick={setFrame} />
        </section>

        <aside aria-label="Details and tasks" className="min-w-0">
          {selected ? (
            <AreaDetail
              key={selected.area.area_id}
              cam={cam}
              camera={camera}
              frame={fi}
              cell={selected}
              resolved={!!resolved[selected.area.area_id]}
              flagged={!!flags[selected.area.area_id]}
              onClose={() => setSel(null)}
              onToggleResolved={() => setResolved(toggle(resolved, selected.area.area_id))}
              onToggleFlag={() => setFlags(toggle(flags, selected.area.area_id))}
            />
          ) : (
            <TaskPanel cells={allCells} filter={filter} tab={tab} resolvedCount={Object.keys(resolved).length} onTab={setTab} onPick={pickCam} />
          )}
        </aside>
      </div>
    </div>
  )
}

// Empty and wrong product weigh 3, half weighs 1
const score = (cells: Cell[]) => count(cells, 'empty') * 3 + count(cells, 'wrong') * 3 + count(cells, 'low')

export default App
