import { cameraName } from '../lib/format'
import { type Cell, count, type Filter, issueParts, STATUS, STATUSES } from '../lib/status'

export type CameraSummary = { id: string; cells: Cell[]; thumb: string }
type Props = { cams: CameraSummary[]; active: string; filter: Filter; onPick: (id: string) => void }

// Cameras sorted by urgency, each with a bar of its area statuses.
export default function CameraList({ cams, active, filter, onPick }: Props) {
  return (
    <nav aria-label="Cameras" className="min-w-0 lg:row-span-2 xl:row-span-1">
      <div className="px-2 pb-2 text-sm font-semibold">Cameras</div>
      <div className="flex gap-1 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible">
        {cams.map((c) => {
          const on = c.id === active
          // With a filter on, cameras that have nothing of that status step back
          const off = !on && filter !== 'all' && count(c.cells, filter) === 0
          return (
            <button
              key={c.id}
              className={`cam flex w-[240px] flex-none items-center gap-3 rounded-[14px] p-2 text-left lg:w-full ${off ? 'off' : ''}`}
              aria-pressed={on}
              aria-label={`${cameraName(c.id)}: ${issueParts(c.cells).join(', ') || 'all in order'}`}
              onClick={() => onPick(c.id)}
            >
              <img className="cam-thumb h-[43px] w-[76px] flex-none rounded-[9px] object-cover" src={c.thumb} alt="" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13.5px] font-medium">{cameraName(c.id)}</span>
                <span className="mt-2.5 flex h-1.5 gap-0.5">
                  {STATUSES.map((st) => {
                    const n = count(c.cells, st)
                    return n ? <span key={st} className="rounded-sm" style={{ flexGrow: n, background: STATUS[st].bar }} /> : null
                  })}
                </span>
              </span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
