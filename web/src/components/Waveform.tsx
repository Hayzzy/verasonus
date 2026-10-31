import type { Segment } from '../lib/api'

/** Mix ash grey (human) into flame orange (AI) by probability. */
export function aiColour(probability: number): string {
  const p = Math.min(1, Math.max(0, probability))
  const mix = (a: number, b: number) => Math.round(a + (b - a) * p)
  return `rgb(${mix(179, 255)} ${mix(179, 122)} ${mix(179, 0)})`
}

/**
 * Bar waveform. With `segments` (and the track's `duration`), each bar is coloured by the AI
 * probability of the segment it falls in: grey for human-sounding, orange for AI-sounding.
 */
export function Waveform({
  peaks,
  segments,
  duration,
}: {
  peaks: number[]
  segments?: Segment[]
  duration?: number
}) {
  const colourAt = (i: number) => {
    if (!segments?.length || !duration) return undefined
    const t = ((i + 0.5) / peaks.length) * duration
    const segment =
      segments.find((s) => t >= s.start_seconds && t < s.end_seconds) ?? segments[segments.length - 1]
    return aiColour(segment.ai_probability)
  }
  return (
    <div
      role="img"
      aria-label={segments ? 'Waveform coloured by AI probability' : 'Waveform of the selected file'}
      className="flex h-16 items-center gap-[2px]"
    >
      {peaks.map((peak, i) => (
        <span
          key={i}
          className="flex-1 rounded-full bg-white/70"
          style={{ height: `${Math.max(6, peak * 100)}%`, backgroundColor: colourAt(i) }}
        />
      ))}
    </div>
  )
}
