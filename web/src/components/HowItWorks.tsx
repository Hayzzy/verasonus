import { AudioWaveform, Cpu, FileMusic, Gauge, SlidersHorizontal } from 'lucide-react'

// The five stages from the "System Design" slide.
const STEPS = [
  {
    icon: FileMusic,
    title: 'Input',
    text: 'A song or a short clip, uploaded or recorded on your phone.',
  },
  {
    icon: SlidersHorizontal,
    title: 'Pre-process',
    text: 'Resampled, cut into short windows, optionally split into stems.',
  },
  {
    icon: AudioWaveform,
    title: 'Features',
    text: 'Spectral features, mel and CQT spectrograms, MERT and CLAP embeddings.',
  },
  {
    icon: Cpu,
    title: 'Models',
    text: 'Classical ML, CNN and Transformer models and embedding probes, in an ensemble.',
  },
  {
    icon: Gauge,
    title: 'Output',
    text: 'A calibrated AI score, a segment timeline, the likely generator and why.',
  },
]

export function HowItWorks() {
  return (
    <section aria-labelledby="how-heading" className="relative pb-24 lg:pb-32">
      <div
        aria-hidden="true"
        className="glow pointer-events-none absolute -bottom-72 left-1/2 -z-10 size-[42rem] -translate-x-1/2 opacity-25 blur-2xl"
      />
      <div className="grid gap-5 lg:grid-cols-2 lg:items-end lg:gap-14">
        <h2
          id="how-heading"
          className="font-display text-4xl font-bold tracking-display sm:text-5xl"
        >
          How VeraSonus listens
        </h2>
        <p className="max-w-xl text-lg leading-relaxed font-light text-white/85">
          From an uploaded track to an explainable verdict in five stages, each one swappable
          as we compare techniques.
        </p>
      </div>

      <ol className="mt-12 grid gap-10 sm:grid-cols-2 lg:grid-cols-5 lg:gap-6">
        {STEPS.map(({ icon: Icon, title, text }, i) => {
          const last = i === STEPS.length - 1
          return (
            <li key={title}>
              <div className="flex items-center gap-4">
                <span className={`ring-badge size-16 shrink-0 ${last ? 'bg-card-warm' : ''}`}>
                  <Icon className="size-7" strokeWidth={1.5} />
                </span>
                {!last && (
                  <span
                    aria-hidden="true"
                    className="relative hidden h-px flex-1 bg-ash after:absolute after:-top-[3.5px] after:-right-px after:border-y-4 after:border-l-[7px] after:border-y-transparent after:border-l-ash lg:block"
                  />
                )}
              </div>
              <p className="eyebrow mt-6">Step {String(i + 1).padStart(2, '0')}</p>
              <h3 className="mt-2 font-display text-2xl font-bold tracking-[-0.03em]">{title}</h3>
              <p className="mt-3 leading-relaxed font-light text-white/80">{text}</p>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
