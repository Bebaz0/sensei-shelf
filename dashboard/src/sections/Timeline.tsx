import { hm } from '../lib/format'
import type { Camera } from '../lib/results'
import { cellsOf, count, STATUS, STATUSES, type Status } from '../lib/status'

type Props = { camera: Camera; frame: number; done: Record<string, boolean>; onPick: (frame: number | null) => void }

const STACK: Status[] = ['ok', 'low', 'wrong', 'empty'] // bottom to top

// One column per capture of the selected camera; clicking one goes back to it.
export default function Timeline({ camera, frame, done, onPick }: Props) {
  const last = camera.captures.length - 1
  const cellsAt = (i: number) => cellsOf(camera, camera.captures[i], i === last ? done : {})
  const current = cellsAt(frame)

  return (
    <div className="rounded-[22px] border border-line bg-surface p-[18px] pb-4">
      {/* What this shelf looks like at the selected capture */}
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 px-1 pb-4">
        <div>
          <div className="mono text-[32px] leading-none font-medium tracking-[-0.03em]">{hm(camera.captures[frame].ts)}</div>
          <div className="mt-1.5 text-xs text-muted">{frame === last ? 'Latest capture' : 'Earlier capture'}</div>
        </div>
        <dl className="grid w-full grid-cols-4 gap-3 sm:w-auto sm:gap-8">
          {STATUSES.map((st) => {
            const n = count(current, st)
            return (
              <div key={st} className="flex flex-col-reverse">
                <dt className="mt-1.5 flex items-center gap-1.5 text-xs text-fg2">
                  <span className="size-2 flex-none rounded-full" style={{ background: STATUS[st].color }} />
                  {STATUS[st].label}
                </dt>
                <dd className="text-[32px] leading-none font-semibold tracking-[-0.03em] tabular-nums" style={{ color: n ? STATUS[st].text : 'var(--color-muted)' }}>
                  {n}
                </dd>
              </div>
            )
          })}
        </dl>
      </div>
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-t border-line px-1 pt-4">
        <span className="text-[15px] font-semibold">Throughout the day</span>
      </div>
      <div className="mt-3.5 overflow-x-auto">
        <div className="grid min-w-[520px] gap-1" style={{ gridTemplateColumns: `repeat(${camera.captures.length}, minmax(0, 1fr))` }}>
          {camera.captures.map((cap, i) => {
            const cells = cellsAt(i)
            const on = i === frame
            return (
              <button
                key={cap.ts}
                className={`mono flex flex-col items-center gap-[7px] rounded-xl border px-[3px] py-[7px] text-[11.5px] transition-colors ${on ? 'border-line3 bg-sel text-fg' : 'border-transparent text-muted hover:bg-hover hover:text-fg2'}`}
                aria-pressed={on}
                aria-label={`Capture at ${hm(cap.ts)}`}
                onClick={() => onPick(i === last ? null : i)}
              >
                <span className="flex h-[58px] w-full flex-col-reverse overflow-hidden rounded-[7px] bg-track">
                  {STACK.map((st) => {
                    const n = count(cells, st)
                    return n ? <span key={st} className="transition-[height] duration-500" style={{ height: `${(n / cells.length) * 100}%`, background: STATUS[st].bar }} /> : null
                  })}
                </span>
                {hm(cap.ts)}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
