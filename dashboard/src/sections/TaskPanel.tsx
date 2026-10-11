import { areaNo, cameraName, plural, pretty } from '../lib/format'
import { type CamCell, type Filter, LEVEL_INDEX, matches, STATUS, STATUSES, type Status } from '../lib/status'
import { Icon } from '../lib/ui'

export type Tab = 'tasks' | 'products'
type Props = {
  cells: CamCell[]
  filter: Filter
  tab: Tab
  resolvedCount: number
  onTab: (t: Tab) => void
  onPick: (cam: string, areaId: string) => void
}

// Task groups, most urgent first; the heading says what to do so rows only need the product and where.
const GROUPS: { st: Status; title: string }[] = [
  { st: 'empty', title: 'Restock now' },
  { st: 'wrong', title: 'Wrong product' },
  { st: 'low', title: 'Top up soon' },
]

// Shown when no area is selected: the to-do list across all cameras, or stock per product.
export default function TaskPanel({ cells, filter, tab, resolvedCount, onTab, onPick }: Props) {
  const groups = GROUPS.filter((g) => matches(filter, g.st))
    .map((g) => ({ ...g, items: cells.filter((c) => c.st === g.st).sort((a, b) => cameraName(a.cam).localeCompare(cameraName(b.cam))) }))
    .filter((g) => g.items.length)

  // Average stock per product across every area it sits in; a wrong product counts as empty.
  // `worst` is the most urgent of those areas, where clicking the product goes.
  const byProduct = new Map<string, { name: string; n: number; sum: number; alerts: number; worst: CamCell }>()
  for (const c of cells) {
    const p = byProduct.get(c.area.product_name) ?? { name: c.area.product_name, n: 0, sum: 0, alerts: 0, worst: c }
    p.n += 1
    p.sum += c.done ? 1 : c.st === 'wrong' ? 0 : LEVEL_INDEX[c.v.stock_level] / 2
    if (c.st !== 'ok') p.alerts += 1
    if (STATUSES.indexOf(c.st) < STATUSES.indexOf(p.worst.st)) p.worst = c
    byProduct.set(p.name, p)
  }
  const products = [...byProduct.values()]
    .filter((p) => p.alerts > 0)
    .map((p) => ({ ...p, avg: p.sum / p.n }))
    .sort((a, b) => a.avg - b.avg)

  return (
    <div className="panel rounded-[22px] border border-line bg-surface p-3">
      <div className="seg w-full" role="group" aria-label="List">
        <span className="seg-ind" style={{ transform: tab === 'products' ? 'translateX(100%)' : 'none' }} />
        <button aria-pressed={tab === 'tasks'} onClick={() => onTab('tasks')}>
          Tasks
        </button>
        <button aria-pressed={tab === 'products'} onClick={() => onTab('products')}>
          By product
        </button>
      </div>

      {tab === 'tasks' ? (
        <div className="mt-2 max-h-[600px] overflow-auto">
          {groups.length === 0 && <p className="px-3 py-8 text-center text-sm text-muted">Nothing to do for this filter</p>}
          {groups.map((g) => (
            <section key={g.st} aria-label={g.title} className="pb-2">
              {/* Heading stays visible while scrolling its group */}
              <h3 className="sticky top-0 z-10 flex items-center gap-2 bg-surface px-3 pt-3 pb-2 text-[13px] font-semibold">
                <span className="size-2.5 rounded-full" style={{ background: STATUS[g.st].color }} />
                {g.title}
                <span className="mono ml-auto text-muted">{g.items.length}</span>
              </h3>
              {g.items.map((c) => (
                <button
                  key={c.area.area_id}
                  className="task flex w-full items-center gap-3 rounded-xl py-2.5 pr-2 pl-3 text-left"
                  onClick={() => onPick(c.cam, c.area.area_id)}
                >
                  <span className="w-1 self-stretch rounded-full" style={{ background: STATUS[c.st].color }} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium" title={pretty(c.area.product_name)}>
                      {pretty(c.area.product_name)}
                    </span>
                    <span className="block truncate text-xs text-muted">
                      {c.st === 'wrong' && c.v.seen_product ? `Has ${pretty(c.v.seen_product)} · ` : ''}
                      {cameraName(c.cam)} · area {areaNo(c.area)}
                    </span>
                  </span>
                  <span className="flex-none text-muted">
                    <Icon name="chevron_right" size={20} />
                  </span>
                </button>
              ))}
            </section>
          ))}
          {resolvedCount > 0 && <p className="px-3 pt-1 pb-2 text-xs text-muted">{plural(resolvedCount, 'area', 'areas')} resolved this shift</p>}
        </div>
      ) : (
        <div className="mt-2 max-h-[600px] overflow-auto">
          <p className="px-3 pt-3 pb-1 text-[13px] text-muted">Average stock per product, across every area it sits in</p>
          {products.map((p) => {
            const pct = `${Math.round(p.avg * 100)}%`
            const color = p.avg < 0.34 ? STATUS.empty.color : p.avg < 0.75 ? STATUS.low.color : STATUS.ok.color
            return (
              <button
                key={p.name}
                className="task flex w-full items-center gap-3 rounded-xl py-3 pr-2 pl-3 text-left"
                aria-label={`${pretty(p.name)}, ${pct} stock, open ${cameraName(p.worst.cam)} area ${areaNo(p.worst.area)}`}
                onClick={() => onPick(p.worst.cam, p.worst.area.area_id)}
              >
                <span className="min-w-0 flex-1">
                  <span className="flex items-baseline justify-between gap-3 text-sm">
                    <span className="min-w-0 truncate font-medium" title={pretty(p.name)}>
                      {pretty(p.name)}
                    </span>
                    <span className="mono flex-none text-fg2">{pct}</span>
                  </span>
                  <span className="mt-2 block h-1.5 overflow-hidden rounded-full bg-track">
                    <span className="block h-full rounded-full transition-[width] duration-500" style={{ width: pct, background: color }} />
                  </span>
                  <span className="mt-1.5 block text-xs text-muted">
                    {p.alerts} of {plural(p.n, 'area', 'areas')} need attention
                  </span>
                </span>
                <span className="flex-none text-muted">
                  <Icon name="chevron_right" size={20} />
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
