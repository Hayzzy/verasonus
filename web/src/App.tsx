import { Backdrop } from './components/Backdrop'
import { Footer } from './components/Footer'
import { Header } from './components/Header'
import { HowItWorks } from './components/HowItWorks'
import { UploadCard } from './components/UploadCard'

const OUTPUTS = ['Calibrated AI score', 'Segment timeline', 'Likely generator', 'Plain-language reasons']

export default function App() {
  return (
    <div className="relative isolate min-h-dvh overflow-clip">
      <Backdrop />
      <Header />
      <main className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <section className="grid items-center gap-10 pt-12 pb-20 sm:pt-16 lg:grid-cols-[1.05fr_1fr] lg:gap-14 lg:pt-24 lg:pb-28">
          <div>
            <p className="eyebrow">AI music detector · prototype</p>
            <h1 className="mt-5 font-display text-5xl leading-[1.04] font-bold tracking-display text-balance sm:text-6xl lg:text-7xl">
              Is this track <span className="text-gradient">AI-generated?</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed font-light text-white/85 sm:text-xl">
              VeraSonus estimates how likely a song is to be AI-generated, shows which parts
              sound synthetic, and explains why, so a person can check the call.
            </p>
            <ul className="mt-8 flex flex-wrap gap-2.5">
              {OUTPUTS.map((output) => (
                <li
                  key={output}
                  className="glass-tile rounded-full border border-ember px-4 py-1.5 text-[0.95rem] text-white/90"
                >
                  {output}
                </li>
              ))}
            </ul>
          </div>
          <UploadCard />
        </section>
        <HowItWorks />
      </main>
      <Footer />
    </div>
  )
}
