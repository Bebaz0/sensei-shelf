import { type Cell, count, type Filter, STATUS, STATUSES } from '../lib/status'

type Props = { cells: Cell[]; filter: Filter; onFilter: (f: Filter) => void }

// One card per status; clicking a card filters the whole page by it.
export default function Kpis({ cells, filter, onFilter }: Props) {
  return (
    <section aria-label="Store summary" className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
      {STATUSES.map((st, i) => {
        const on = filter === st
        return (
          <button
            key={st}
            className={`kpi flex min-h-[132px] flex-col gap-1.5 rounded-[20px] border px-5 pt-[18px] pb-4 text-left ${on ? 'border-sel-line bg-sel' : 'border-line bg-surface hover:border-line3'}`}
            style={{ animationDelay: `${i * 80}ms` }}
            aria-pressed={on}
            onClick={() => onFilter(on ? 'all' : st)}
          >
            <span className="flex w-full items-center justify-between gap-2 text-[13.5px] text-fg2">
              {STATUS[st].label}
              <span className="size-2.5 flex-none rounded-full" style={{ background: STATUS[st].color }} />
            </span>
            <span className="mt-1.5 text-[40px] leading-none font-medium tracking-[-0.035em] tabular-nums sm:text-[48px]" style={{ color: STATUS[st].text }}>
              {count(cells, st)}
            </span>
          </button>
        )
      })}
    </section>
  )
}
