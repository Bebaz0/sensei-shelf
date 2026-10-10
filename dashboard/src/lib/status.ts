import type { Area, Camera, Capture, StockLevel, Verdict } from './results'

export type Status = 'empty' | 'wrong' | 'low' | 'ok'
export type Filter = Status | 'all'
export type Cell = { area: Area; v: Verdict; st: Status; done: boolean }
export type CamCell = Cell & { cam: string }

// Most urgent first; also the order of KPIs, legend and task sorting.
export const STATUSES: Status[] = ['empty', 'wrong', 'low', 'ok']

export const STATUS: Record<Status, { label: string; color: string; text: string; bar: string }> = {
  empty: { label: 'Empty', color: 'var(--color-empty)', text: 'var(--color-empty)', bar: 'var(--color-empty)' },
  wrong: { label: 'Wrong product', color: 'var(--color-wrong)', text: 'var(--color-wrong)', bar: 'var(--color-wrong)' },
  low: { label: 'Half', color: 'var(--color-low)', text: 'var(--color-low-text)', bar: 'var(--color-low)' },
  ok: { label: 'In order', color: 'var(--color-ok)', text: 'var(--color-fg)', bar: 'var(--color-okbar)' },
}

export const LEVEL_INDEX: Record<StockLevel, number> = { empty: 0, half: 1, full: 2 }

// A wrong product outranks the stock level: the shelf is wrong whatever its fill.
export function statusOf(v: Verdict): Status {
  if (!v.planogram_compliant) return 'wrong'
  return v.stock_level === 'empty' ? 'empty' : v.stock_level === 'half' ? 'low' : 'ok'
}

// Areas resolved this shift (`done`) count as in order. Pass {} for anything but the latest capture.
export const cellsOf = (camera: Camera, capture: Capture, done: Record<string, boolean>): Cell[] =>
  camera.areas.map((area) => {
    const v = capture.verdicts[area.area_id]
    const isDone = !!done[area.area_id] && statusOf(v) !== 'ok'
    return { area, v, st: isDone ? 'ok' : statusOf(v), done: isDone }
  })

export const count = (cells: Cell[], st: Status) => cells.filter((c) => c.st === st).length
export const matches = (filter: Filter, st: Status) => filter === 'all' || filter === st

// "2 empty", "1 wrong", "3 half" for the statuses that need attention
export const issueParts = (cells: Cell[]) =>
  (['empty', 'wrong', 'low'] as const)
    .map((st) => [count(cells, st), st === 'low' ? 'half' : st] as const)
    .filter(([n]) => n)
    .map(([n, label]) => `${n} ${label}`)
