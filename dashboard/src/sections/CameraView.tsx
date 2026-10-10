import { useState } from 'react'
import { areaNo, cameraName, hm, pretty } from '../lib/format'
import { centroid, clamp, schemaGeometry } from '../lib/geometry'
import type { Capture } from '../lib/results'
import { type Cell, type Filter, matches, STATUS } from '../lib/status'

type Props = { cam: string; capture: Capture; cells: Cell[]; sel: string | null; filter: Filter; onSelect: (areaId: string) => void }

// Selected camera: title, Photo/Schema toggle, and the stage with one pin per area.
export default function CameraView({ cam, capture, cells, sel, filter, onSelect }: Props) {
  const [view, setView] = useState<'photo' | 'schema'>('photo')
  const geo = schemaGeometry(cells.map((c) => c.area))
  const state = (c: Cell) => `${c.st} ${c.area.area_id === sel ? 'sel' : sel ? 'unsel' : ''} ${matches(filter, c.st) ? '' : 'dim'}`

  return (
    <>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="mono text-xs tracking-[.08em] text-muted">
            CAMERA {cam}
          </div>
          <h2 className="mt-1 text-[26px] font-semibold tracking-[-0.025em] sm:text-[30px]">{cameraName(cam)}</h2>
        </div>
        <div className="seg" role="group" aria-label="Camera view">
          <span className="seg-ind" style={{ transform: view === 'schema' ? 'translateX(100%)' : 'none' }} />
          <button aria-pressed={view === 'photo'} onClick={() => setView('photo')}>
            Photo
          </button>
          <button aria-pressed={view === 'schema'} onClick={() => setView('schema')}>
            Schema
          </button>
        </div>
      </div>

      {/* Remounting on camera, capture or view change replays the entry animations */}
      <div
        key={`${cam}-${capture.ts}-${view}`}
        className={`relative overflow-hidden rounded-[22px] border border-line ${view === 'photo' ? 'photo aspect-video bg-black' : 'aspect-[16/10] bg-surface2'}`}
      >
        {view === 'photo' ? (
          <>
            <img className="stage-img absolute inset-0 size-full object-cover" src={capture.image} alt={`Camera ${cam} at ${hm(capture.ts)}`} />
            <svg viewBox="0 0 1280 720" preserveAspectRatio="none" className="absolute inset-0 size-full">
              {cells.map((c) => (
                <polygon
                  key={c.area.area_id}
                  points={c.area.polygon.map((p) => p.join(',')).join(' ')}
                  className={`poly ${state(c)}`}
                  style={{ stroke: STATUS[c.st].color, fill: STATUS[c.st].color }}
                  onClick={() => onSelect(c.area.area_id)}
                />
              ))}
            </svg>
          </>
        ) : (
          <svg viewBox="0 0 320 200" className="absolute inset-0 size-full" role="img" aria-label={`Schema of ${cameraName(cam)}`}>
            <defs>
              <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="7" height="7" fill="var(--color-wrong)" fillOpacity={0.16} />
                <rect width="2.4" height="7" fill="var(--color-wrong)" />
              </pattern>
              {cells.map((c, i) => (
                <clipPath key={c.area.area_id} id={`clip-${c.area.area_id}`}>
                  <polygon points={geo[i].points} />
                </clipPath>
              ))}
            </defs>
            {cells.map((c, i) => {
              const g = geo[i]
              // Fill height = stock level
              const lv = c.done ? 1 : c.st === 'empty' ? 0 : c.st === 'low' || c.v.stock_level === 'half' ? 0.5 : 1
              return (
                <g key={c.area.area_id} className={`transition-opacity duration-300 ${matches(filter, c.st) ? '' : 'dim'}`}>
                  <polygon points={g.points} fill="var(--color-cell)" />
                  <g clipPath={`url(#clip-${c.area.area_id})`}>
                    <rect
                      className="liq"
                      x={g.bx}
                      y={g.by}
                      width={g.bw}
                      height={g.bh}
                      style={{
                        transform: `scaleY(${lv})`,
                        fill: c.st === 'wrong' ? 'url(#hatch)' : c.st === 'low' ? 'var(--color-low)' : 'var(--color-ok)',
                        animationDelay: `${120 + i * 60}ms`,
                      }}
                    />
                  </g>
                  <polygon
                    className={c.st === 'empty' ? 'outline-empty' : ''}
                    points={g.points}
                    fill="none"
                    stroke={c.st === 'ok' ? 'var(--color-line3)' : STATUS[c.st].color}
                    strokeWidth={c.st === 'ok' ? 0.8 : 1.5}
                    strokeDasharray={c.st === 'empty' ? '3 2.5' : undefined}
                    strokeLinejoin="round"
                  />
                  {c.area.area_id === sel && <polygon className="ring" points={g.points} />}
                  <polygon className="hit" points={g.points} onClick={() => onSelect(c.area.area_id)} />
                </g>
              )
            })}
          </svg>
        )}

        {cells.map((c, i) => {
          // Pin position in % of the stage: image is 1280x720, schema is 320x200
          const [x, y] = view === 'photo' ? centroid(c.area.polygon).map((v, k) => v / (k ? 7.2 : 12.8)) : [geo[i].cx / 3.2, geo[i].cy / 2]
          return (
            <button
              key={c.area.area_id}
              className={`pin ${state(c)}`}
              style={{ left: `${clamp(x, 4, 96)}%`, top: `${clamp(y, 7, 93)}%`, animationDelay: `${150 + i * 50}ms` }}
              aria-label={`Area ${areaNo(c.area)}: ${pretty(c.area.product_name)}, ${STATUS[c.st].label}`}
              onClick={() => onSelect(c.area.area_id)}
            >
              {areaNo(c.area)}
            </button>
          )
        })}
      </div>
    </>
  )
}
