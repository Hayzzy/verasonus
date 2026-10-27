import { useId } from 'react'

// (x centre, height) on a 100-unit grid; matches public/favicon.svg and scripts/make_icons.py
const BARS = [
  [33, 14],
  [41.5, 28],
  [50, 40],
  [58.5, 24],
  [67, 12],
] as const

/** VeraSonus mark: the slides' orange ring around five waveform bars. */
export function Logo({ className }: { className?: string }) {
  const gradientId = `vs-ring-${useId().replace(/[^\w-]/g, '')}`
  return (
    <svg viewBox="0 0 100 100" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#FF7A00" />
          <stop offset="1" stopColor="#D56A46" />
        </linearGradient>
      </defs>
      <circle cx="50" cy="50" r="34" fill="none" stroke={`url(#${gradientId})`} strokeWidth="6" />
      <g fill="#fff">
        {BARS.map(([x, height]) => (
          <rect key={x} x={x - 2.75} y={50 - height / 2} width="5.5" height={height} rx="2.75" />
        ))}
      </g>
    </svg>
  )
}
