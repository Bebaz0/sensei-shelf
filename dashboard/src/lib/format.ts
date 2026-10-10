import type { Area } from './results'

// Zone names are a UI concern; the pipeline only knows camera ids.
const CAMERA_NAMES: Record<string, string> = {
  '6215': 'Banana island',
  '6371': 'Watermelons and melons',
  '6373': 'Apples and pears',
  '6376': 'Citrus and kiwis',
  '6383': 'Vegetables, cold wall',
  '6385': 'Fuji apples',
  '6386': 'Cabbages, cold wall',
  '6443': 'Tomatoes',
  '6462': 'Grapes and berries',
  '6468': 'Tropical fruit',
  '6491': 'Stone fruit',
  '6591': 'Potatoes',
  '6594': 'Onions and garlic',
  '6781': 'Potato island',
}
export const cameraName = (id: string) => CAMERA_NAMES[id] ?? `Camera ${id}`

// "6215_7" -> "7"
export const areaNo = (area: Area) => area.area_id.split('_')[1]

export const hm = (ts: number) =>
  new Date(ts * 1000).toLocaleTimeString('en-GB', { timeZone: 'Europe/Lisbon', hour: '2-digit', minute: '2-digit' })

// Planogram names are all caps ("BANANA CAT I CNT KG"); sentence case reads better.
export const pretty = (name: string) => name.charAt(0) + name.slice(1).toLowerCase()

export const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`
