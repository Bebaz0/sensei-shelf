// Google Material Symbols, subset loaded in index.html (add new names to its icon_names list).
export const Icon = ({ name, size = 18 }: { name: string; size?: number }) => (
  <span className="material-symbols-rounded" style={{ fontSize: size }} aria-hidden>
    {name}
  </span>
)

export const btn = 'h-[46px] rounded-[14px] border px-4 text-sm font-medium transition-colors active:scale-[.98]'
