/** Bar waveform. In Phase 3 each bar is coloured by the AI probability of its segment. */
export function Waveform({ peaks }: { peaks: number[] }) {
  return (
    <div role="img" aria-label="Waveform of the selected file" className="flex h-16 items-center gap-[2px]">
      {peaks.map((peak, i) => (
        <span
          key={i}
          className="flex-1 rounded-full bg-white/70"
          style={{ height: `${Math.max(6, peak * 100)}%` }}
        />
      ))}
    </div>
  )
}
