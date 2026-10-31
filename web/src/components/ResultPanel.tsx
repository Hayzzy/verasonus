import { Bot, CircleHelp, Layers, User } from 'lucide-react'

import type { AnalysisResult, Segment } from '../lib/api'
import { formatDuration } from '../lib/audio'
import { aiColour } from './Waveform'

const VERDICTS = {
  likely_ai: { label: 'Likely AI-generated', icon: Bot },
  partly_ai: { label: 'Partly AI-generated', icon: Layers },
  uncertain: { label: 'Uncertain', icon: CircleHelp },
  likely_human: { label: 'Likely human-made', icon: User },
} as const

const GENERATORS: Record<string, string> = {
  audioldm2: 'AudioLDM 2',
  musicgen: 'MusicGen',
  musicldm: 'MusicLDM',
  mustango: 'Mustango',
  stable_audio_open: 'Stable Audio Open',
  suno: 'Suno',
  udio: 'Udio',
  unknown: 'Unknown generator',
}

const STEMS = { vocals: 'Vocals', instrumental: 'Instrumental' } as const

/** Whole percent, but never a flat 0% or 100%: the models are not that certain. */
export function percent(p: number): string {
  if (p < 0.01) return '<1%'
  if (p > 0.99) return '>99%'
  return `${Math.round(p * 100)}%`
}

/** The analysis: verdict and score, stem scores, timeline, likely generator and reasons. */
export function ResultPanel({ result }: { result: AnalysisResult }) {
  const verdict = VERDICTS[result.verdict]
  const Icon = verdict.icon
  const strongest = Math.max(...result.explanation.top_features.map((f) => Math.abs(f.contribution)), 1e-6)
  return (
    <div className="mt-5 space-y-5" aria-live="polite">
      <div className="flex items-center gap-4">
        <span className="ring-badge glass-tile size-14 shrink-0">
          <Icon className="size-6 text-apricot" strokeWidth={1.6} />
        </span>
        <div>
          <p className="font-display text-2xl font-bold tracking-display">{verdict.label}</p>
          <p className="text-mist">
            AI probability{' '}
            <span className="font-display text-xl font-bold text-gradient">
              {percent(result.ai_probability)}
            </span>
          </p>
        </div>
      </div>

      {(result.stems ?? []).length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {result.stems.map((stem) => (
            <div key={stem.name} className="glass-tile rounded-xl border border-ember/60 px-4 py-3">
              <p className="eyebrow">{STEMS[stem.name]}</p>
              <p className="mt-1 font-display text-2xl font-bold text-gradient">
                {percent(stem.ai_probability)}
              </p>
              <p className="text-xs text-ash">AI probability of the separated {stem.name}</p>
            </div>
          ))}
        </div>
      )}

      <p className="text-[0.95rem] leading-relaxed text-white/85">{result.explanation.summary}</p>

      {/* A generator guess only means something when the whole track sounds AI-made. */}
      {result.generator && result.verdict === 'likely_ai' && (
        <p className="text-sm text-mist">
          Closest known generator:{' '}
          <span className="font-semibold text-white">
            {GENERATORS[result.generator.name] ?? result.generator.name}
          </span>
          {result.generator.name !== 'unknown' && ` (${percent(result.generator.probability)})`}
        </p>
      )}

      {result.segments.length > 1 && (
        <Timeline segments={result.segments} duration={result.duration_seconds} />
      )}

      {result.explanation.top_features.length > 0 && (
        <div>
          <p className="eyebrow">What drove the score</p>
          <ul className="mt-2 space-y-1.5">
            {result.explanation.top_features.map((f, i) => (
              <li key={i} className="grid grid-cols-[1fr_6rem] items-center gap-3 text-sm">
                <span className="truncate text-white/85">{f.name}</span>
                <span className="flex h-2 rounded-full bg-white/10">
                  <span
                    className={`h-2 rounded-full ${f.contribution > 0 ? 'bg-flame' : 'bg-mist'}`}
                    style={{ width: `${(Math.abs(f.contribution) / strongest) * 100}%` }}
                  />
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-ash">Orange pushes towards AI, grey towards human.</p>
        </div>
      )}

      <p className="text-xs leading-relaxed text-ash">
        Trained on Suno, Udio and five research generators (MusicGen, Stable Audio Open and
        others). Newer or unlisted generators and heavily edited tracks can be missed. A score
        is a hint, not proof. Model {result.model_version}.
      </p>
    </div>
  )
}

/** One bar across the track, each segment coloured by its AI probability. */
function Timeline({ segments, duration }: { segments: Segment[]; duration: number }) {
  const peak = segments.reduce((a, b) => (b.ai_probability > a.ai_probability ? b : a))
  return (
    <div>
      <p className="eyebrow">Timeline</p>
      <div
        role="img"
        aria-label="AI probability over the track"
        className="mt-2 flex h-4 overflow-hidden rounded-full border border-ash/60"
      >
        {segments.map((s) => (
          <span
            key={s.start_seconds}
            title={`${formatDuration(s.start_seconds)}–${formatDuration(s.end_seconds)}: ${percent(s.ai_probability)}`}
            style={{
              width: `${((s.end_seconds - s.start_seconds) / duration) * 100}%`,
              backgroundColor: aiColour(s.ai_probability),
            }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-xs text-ash">
        <span>0:00</span>
        <span>
          Most AI-like: {formatDuration(peak.start_seconds)}–{formatDuration(peak.end_seconds)} (
          {percent(peak.ai_probability)})
        </span>
        <span>{formatDuration(duration)}</span>
      </div>
    </div>
  )
}
