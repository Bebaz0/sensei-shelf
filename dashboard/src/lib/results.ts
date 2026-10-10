import planogram from '../../../data/planogram.json'

export type StockLevel = 'empty' | 'half' | 'full'

// out/results.json as defined in CONTRACTS.md: {camera_id: {filename: {area_id: AreaResult}}}
type AreaResult = {
  pred: { planogram_compliant: boolean; stock_level: StockLevel; stock_value: number }
  p_stock: Record<StockLevel, number>
  p_wrong: number
  observed_product_id: string | null
  low_confidence: boolean
  reason: string
  crop: string
}
type ResultsFile = Record<string, Record<string, Record<string, AreaResult>>>

// The page's view of it: per camera, the planogram areas plus every capture oldest first.
export type Verdict = { planogram_compliant: boolean; stock_level: StockLevel; seen_product: string | null }
export type Area = { area_id: string; product_name: string; polygon: [number, number][] }
export type Capture = { image: string; ts: number; verdicts: Record<string, Verdict> }
export type Camera = { areas: Area[]; captures: Capture[] }
export type Results = Record<string, Camera>

const BASE = import.meta.env.BASE_URL
const CAMERAS = planogram['Frutas e Legumes'].cameras
const PLANOGRAM: Record<string, (typeof CAMERAS)['6215']> = CAMERAS
const PRODUCT_NAMES = new Map(Object.values(PLANOGRAM).flatMap((c) => c.products.map((p) => [p.product_id, p.product_name])))

// Filled by scripts/sync_dashboard.sh; in dev, vite.config.ts falls back to mocks/results.json until then.
export async function loadResults(): Promise<Results> {
  const res = await fetch(`${BASE}data/results.json`)
  if (!res.ok) throw new Error(`Could not load results: ${res.status}`)
  const file: ResultsFile = await res.json()
  return Object.fromEntries(
    Object.entries(file).map(([cam, frames]) => {
      const areas = (PLANOGRAM[cam]?.products ?? []).map((p) => {
        const c = p.image_coordinates
        return { area_id: p.area_id, product_name: p.product_name, polygon: [c.top_left, c.top_right, c.bottom_right, c.bottom_left] as [number, number][] }
      })
      const captures = Object.entries(frames)
        .map(([filename, byArea]) => ({
          image: `${BASE}data/frames/${cam}/${filename}`,
          ts: parseInt(filename), // filenames are <unix_ts>.jpg
          verdicts: Object.fromEntries(
            Object.entries(byArea).map(([id, a]) => [
              id,
              {
                planogram_compliant: a.pred.planogram_compliant,
                stock_level: a.pred.stock_level,
                seen_product: a.observed_product_id ? (PRODUCT_NAMES.get(a.observed_product_id) ?? a.observed_product_id) : null,
              },
            ]),
          ),
        }))
        .sort((a, b) => a.ts - b.ts)
      return [cam, { areas, captures }]
    }),
  )
}
