import { useEffect, useState } from 'react'
import { loadResults, type Results } from './lib/results'
import { type Cell, cellsOf, count, type Filter } from './lib/status'
import AreaDetail from './sections/AreaDetail'
import CameraList from './sections/CameraList'
import CameraView, { type View } from './sections/CameraView'
import Header from './sections/Header'
import Kpis from './sections/Kpis'
import Overview from './sections/Overview'
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

// Last photo/schema choice, so a reload keeps it like the theme does.
function readView(): View {
  try {
    return localStorage.getItem('view') === 'schema' ? 'schema' : 'photo'
  } catch {
    return 'photo'
  }
}

function App() {
  const [data, setData] = useState<Results | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [camId, setCamId] = useState<string | null>(null) // null = store overview
  const [frame, setFrame] = useState<number | null>(null) // capture index, null = latest
  const [sel, setSel] = useState<string | null>(null) // selected area_id
  const [filter, setFilter] = useState<Filter>('all')
  const [camFilter, setCamFilter] = useState<Filter>('all') // timeline chips, open camera only
  const [view, setView] = useState<View>(readView) // photo or schema, kept across cameras and captures until the user toggles it
  const [tab, setTab] = useState<Tab>('tasks')
  const [resolved, setResolved] = useState<Flags>({}) // resolved this shift, local only for now
  const [flags, setFlags] = useState<Flags>({}) // reported as wrong detection, local only for now

  useEffect(() => {
    try {
      localStorage.setItem('view', view)
    } catch {
      // storage blocked: view just won't persist across reloads
    }
  }, [view])

  // A click on empty space peels one layer: area, camera filter, store filter, then the open camera.
  // Re-subscribed every render so it sees the current state.
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if ((e.target as Element).closest('button, polygon, .panel')) return
      if (sel) setSel(null)
      else if (camFilter !== 'all') setCamFilter('all')
      else if (filter !== 'all') setFilter('all')
      else setCamId(null)
    }
    document.addEventListener('click', onClick)
    return () => document.removeEventListener('click', onClick)
  })

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

  // Open camera and capture, if any
  const camera = camId ? data[camId] : null
  const last = camera ? camera.captures.length - 1 : 0
  const fi = frame ?? last
  const capture = camera?.captures[fi]
  const cells = camera && capture ? cellsOf(camera, capture, fi === last ? resolved : {}) : []
  const selected = cells.find((c) => c.area.area_id === sel)

  const pickCam = (id: string, areaId: string | null = null) => {
    setCamId(id)
    setFrame(null)
    setSel(areaId)
    if (id !== camId) setCamFilter('all')
  }

  return (
    <div className="mx-auto flex max-w-[1480px] flex-col gap-[22px] px-4 pt-6 pb-10 sm:px-7">
      <Header />
      <Kpis cells={allCells} filter={filter} onFilter={setFilter} />

      <div className="grid items-start gap-[22px] lg:grid-cols-[250px_minmax(0,1fr)] xl:grid-cols-[262px_minmax(0,1fr)_340px]">
        <CameraList cams={cams} active={camId} filter={filter} onPick={pickCam} />

        <section aria-label={camId ? 'Selected camera' : 'All cameras'} className="flex min-w-0 flex-col gap-4">
          {camId && camera && capture ? (
            <>
              <CameraView
                cam={camId}
                capture={capture}
                view={view}
                onView={setView}
                cells={cells}
                sel={sel}
                filter={camFilter !== 'all' ? camFilter : filter}
                onSelect={setSel}
              />
              <Timeline camera={camera} frame={fi} done={resolved} filter={camFilter} onPick={setFrame} onFilter={setCamFilter} />
            </>
          ) : (
            <Overview cams={cams} filter={filter} onPick={pickCam} />
          )}
        </section>

        <aside aria-label="Details and tasks" className="min-w-0">
          {camId && camera && selected ? (
            <AreaDetail
              key={selected.area.area_id}
              cam={camId}
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
            <TaskPanel cells={camId ? allCells.filter((c) => c.cam === camId) : allCells} filter={filter} tab={tab} resolvedCount={Object.keys(resolved).length} onTab={setTab} onPick={pickCam} />
          )}
        </aside>
      </div>
    </div>
  )
}

// Empty and wrong product weigh 3, half weighs 1
const score = (cells: Cell[]) => count(cells, 'empty') * 3 + count(cells, 'wrong') * 3 + count(cells, 'low')

export default App
