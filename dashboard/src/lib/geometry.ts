import type { Area } from './results'

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export const centroid = (poly: [number, number][]) => [
  poly.reduce((s, p) => s + p[0], 0) / poly.length,
  poly.reduce((s, p) => s + p[1], 0) / poly.length,
]

// Fit all area polygons of a camera into a 320x200 box for the schematic view.
export function schemaGeometry(areas: Area[]) {
  const r1 = (v: number) => Math.round(v * 10) / 10
  const polys = areas.map((a) => a.polygon.map(([x, y]) => [clamp(x, 0, 1280), clamp(y, 0, 720)]))
  const xs = polys.flat().map((p) => p[0])
  const ys = polys.flat().map((p) => p[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const sc = Math.min(296 / (x1 - x0), 176 / (y1 - y0))
  const ox = (320 - (x1 - x0) * sc) / 2 - x0 * sc
  const oy = (200 - (y1 - y0) * sc) / 2 - y0 * sc
  return polys.map((p) => {
    const np = p.map(([x, y]) => [r1(x * sc + ox), r1(y * sc + oy)])
    const nx = np.map((q) => q[0])
    const ny = np.map((q) => q[1])
    const bx = Math.min(...nx)
    const by = Math.min(...ny)
    return {
      points: np.map((q) => q.join(',')).join(' '),
      bx,
      by,
      bw: Math.max(...nx) - bx,
      bh: Math.max(...ny) - by,
      cx: nx.reduce((s, v) => s + v, 0) / nx.length,
      cy: ny.reduce((s, v) => s + v, 0) / ny.length,
    }
  })
}
