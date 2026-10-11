import { cameraName } from '../lib/format'
import { count, type Filter, issueParts } from '../lib/status'
import { type CameraSummary, StatusBar } from './CameraList'

type Props = { cams: CameraSummary[]; filter: Filter; onPick: (id: string) => void }

// Default middle view while no camera is open: every camera's latest capture, most urgent first.
export default function Overview({ cams, filter, onPick }: Props) {
  return (
    <>
      <div>
        <div className="mono text-xs tracking-[.08em] text-muted">STORE</div>
        <h2 className="mt-1 text-[26px] font-semibold tracking-[-0.025em] sm:text-[30px]">All cameras</h2>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {cams.map((c, i) => {
          const off = filter !== 'all' && count(c.cells, filter) === 0
          const issues = issueParts(c.cells).join(', ')
          return (
            <button
              key={c.id}
              className={`cam tile flex min-w-0 flex-col gap-2.5 rounded-[18px] border border-line bg-surface p-2 pb-3 text-left ${off ? 'off' : ''}`}
              style={{ animationDelay: `${i * 40}ms` }}
              aria-label={`${cameraName(c.id)}: ${issues || 'all in order'}`}
              onClick={() => onPick(c.id)}
            >
              <span className="block aspect-video overflow-hidden rounded-[12px] bg-black">
                <img className="cam-thumb size-full object-cover" src={c.thumb} alt="" />
              </span>
              <span className="min-w-0 px-1.5">
                <span className="block truncate text-[13.5px] font-medium">{cameraName(c.id)}</span>
                <StatusBar cells={c.cells} />
                <span className="mt-2 block truncate text-xs text-muted">{issues || 'All in order'}</span>
              </span>
            </button>
          )
        })}
      </div>
    </>
  )
}
