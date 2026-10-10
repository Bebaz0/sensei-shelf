import { useEffect, useState } from 'react'
import { hm } from '../lib/format'
import { Icon } from '../lib/ui'

type Theme = 'light' | 'dark'

function readTheme(): Theme {
  try {
    return localStorage.getItem('theme') === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export default function Header({ updated }: { updated: number }) {
  const [theme, setTheme] = useState<Theme>(readTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('theme', theme)
    } catch {
      // storage blocked: theme just won't persist
    }
  }, [theme])

  return (
    <header className="flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3.5">
        <svg width="38" height="38" viewBox="0 0 38 38" aria-hidden>
          <rect width="38" height="38" rx="11" fill="var(--color-fg)" />
          <path d="M8 19c3.4-6 7.2-9 11-9s7.6 3 11 9c-3.4 6-7.2 9-11 9s-7.6-3-11-9z" fill="none" stroke="var(--color-bg)" strokeWidth="2.4" />
          <circle cx="19" cy="19" r="4" fill="var(--color-bg)" />
        </svg>
        <div>
          <h1 className="text-lg font-semibold tracking-[-0.01em]">Sensei Shelf</h1>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2.5">
        <span className="mono inline-flex h-[34px] items-center rounded-full border border-line bg-surface px-3.5 text-[12.5px]">Updated {hm(updated)}</span>
        <div className="seg" role="group" aria-label="Theme">
          <span className="seg-ind" style={{ transform: theme === 'dark' ? 'translateX(100%)' : 'none' }} />
          <button aria-pressed={theme === 'light'} onClick={() => setTheme('light')}>
            <Icon name="light_mode" />
            Light
          </button>
          <button aria-pressed={theme === 'dark'} onClick={() => setTheme('dark')}>
            <Icon name="dark_mode" />
            Dark
          </button>
        </div>
      </div>
    </header>
  )
}
