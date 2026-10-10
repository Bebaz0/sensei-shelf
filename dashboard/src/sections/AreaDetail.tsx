import { cameraName, hm, pretty } from '../lib/format'
import type { Camera } from '../lib/results'
import { type Cell, LEVEL_INDEX, STATUS, type Status, statusOf } from '../lib/status'
import { btn, Icon } from '../lib/ui'

type Props = {
  cam: string
  camera: Camera
  frame: number
  cell: Cell
  resolved: boolean
  flagged: boolean
  onClose: () => void
  onToggleResolved: () => void
  onToggleFlag: () => void
}

const METER: Status[] = ['empty', 'low', 'ok'] // colour of each stock level

// The selected area: what it should be, what the camera sees, what to do about it.
export default function AreaDetail({ cam, camera, frame, cell, resolved, flagged, onClose, onToggleResolved, onToggleFlag }: Props) {
  const id = cell.area.area_id
  const last = camera.captures.length - 1
  const isLatest = frame === last
  const product = pretty(cell.area.product_name)
  const seen = cell.v.seen_product ? pretty(cell.v.seen_product) : null
  const lit = cell.done ? 2 : LEVEL_INDEX[cell.v.stock_level]

  return (
    <div className="panel flex flex-col gap-[18px] rounded-[22px] border border-line bg-surface p-[22px]">
      <div className="flex items-center justify-between">
        <span
          className="inline-flex h-[30px] items-center rounded-full px-3 text-[13px] font-semibold"
          style={{ background: STATUS[cell.st].color, color: cell.st === 'low' ? '#121417' : cell.st === 'ok' ? '#ffffff' : 'var(--color-on-strong)' }}
        >
          {STATUS[cell.st].label}
        </span>
        <button className="grid size-11 place-items-center rounded-xl border border-line text-fg2 transition-colors hover:bg-hover" aria-label="Close details" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>

      <div>
        <div className="mono text-xs tracking-[.08em] text-muted">
          AREA {id} · {cameraName(cam).toUpperCase()}
        </div>
        <h2 className="mt-1.5 text-[26px] leading-[1.15] font-semibold tracking-[-0.025em]">{product}</h2>
      </div>

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        {/*<dt className="text-muted">Planogram</dt>
        <dd>{product}</dd>
        <dt className="text-muted">Camera sees</dt>
        <dd style={{ color: cell.st === 'wrong' || cell.st === 'empty' ? STATUS[cell.st].color : undefined }}>
          {cell.st === 'wrong' ? (seen ?? 'Another product') : cell.st === 'empty' ? 'Nothing, the area is empty' : product}
        </dd>*/}
        <dt className="text-muted">Capture</dt>
        <dd className="mono">{hm(camera.captures[frame].ts)}</dd>
      </dl>

      <div>
        <div className="flex gap-1.5">
          {METER.map((st, i) => (
            <span key={st} className="h-2.5 flex-1 rounded-[5px] transition-colors" style={{ background: i === lit ? STATUS[st].color : 'var(--color-track)' }} />
          ))}
        </div>
        <div className="mt-2 flex justify-between text-xs text-muted">
          <span>Empty</span>
          <span>Half</span>
          <span>Full</span>
        </div>
      </div>

      <div>
        <div className="mb-2.5 text-[13px] text-muted">This area today</div>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${camera.captures.length}, minmax(0, 1fr))` }}>
          {camera.captures.map((cap, i) => {
            const st = i === last && resolved ? 'ok' : statusOf(cap.verdicts[id])
            return <span key={cap.ts} title={`${hm(cap.ts)} · ${STATUS[st].label}`} className="h-[22px] rounded-md" style={{ background: STATUS[st].bar }} />
          })}
        </div>
        <div className="mono mt-1.5 flex justify-between text-[11.5px] text-muted">
          <span>{hm(camera.captures[0].ts)}</span>
          <span>{hm(camera.captures[last].ts)}</span>
        </div>
      </div>

      <div className="grid gap-2">
        {isLatest && cell.done && (
          <button className={`${btn} border-line2 hover:bg-hover`} onClick={onToggleResolved}>
            Resolved this shift · undo
          </button>
        )}
        <button className={`${btn} border-line2 hover:bg-hover`} onClick={onToggleFlag}>
          {flagged ? 'Reported, thanks' : 'Detection is wrong'}
        </button>
      </div>
    </div>
  )
}
