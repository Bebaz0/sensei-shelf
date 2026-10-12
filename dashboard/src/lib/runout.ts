// "Runs empty at ~HH:MM": straight-line extrapolation of an area's expected fill since its last restock.
// fill = expected stock value from the model's probabilities (empty 0, half 0.5, full 1).

export type Point = { ts: number; fill: number }

export const EMPTY_FILL = 0.25 // halfway between empty (0) and half (0.5)
const RESTOCK_JUMP = 0.25 // a rise this big between two captures starts a new depletion run
const MIN_POINTS = 3
const WINDOW = 4 // only the latest captures of the run: the recent rate, not the whole day's
const MIN_DROP = 0.15 // a smaller fall over the window is model noise, not depletion
const MAX_AHEAD_S = 6 * 3600 // further than this is not a useful shift estimate

// Unix time at which the area is expected to reach EMPTY_FILL, or null when it is not running down
// (too few captures since the last restock, flat, rising or noise-sized trend, already empty, or too far ahead).
export function runsEmptyAt(points: Point[]): number | null {
  let start = 0
  for (let i = 1; i < points.length; i++) if (points[i].fill - points[i - 1].fill >= RESTOCK_JUMP) start = i
  const run = points.slice(start).slice(-WINDOW)
  if (run.length < MIN_POINTS) return null
  const now = run[run.length - 1]
  if (now.fill <= EMPTY_FILL || run[0].fill - now.fill < MIN_DROP) return null
  const mt = run.reduce((s, p) => s + p.ts, 0) / run.length
  const mf = run.reduce((s, p) => s + p.fill, 0) / run.length
  const sxx = run.reduce((s, p) => s + (p.ts - mt) ** 2, 0)
  const slope = run.reduce((s, p) => s + (p.ts - mt) * (p.fill - mf), 0) / sxx // fill per second
  if (!(slope < 0)) return null
  const at = mt + (EMPTY_FILL - mf) / slope
  return at > now.ts && at - now.ts <= MAX_AHEAD_S ? at : null
}
