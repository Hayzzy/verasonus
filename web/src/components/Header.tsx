import { Logo } from './Logo'

export function Header() {
  return (
    <header className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 pt-[max(1.25rem,env(safe-area-inset-top))] sm:px-6 lg:px-8">
      <a
        href="/"
        className="flex items-center gap-2.5 rounded-full focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-apricot"
      >
        <Logo className="size-9" />
        <span className="font-display text-xl font-bold tracking-[-0.03em]">VeraSonus</span>
      </a>
      <span className="rounded-full border border-ember/80 bg-ink/50 px-3.5 py-1 text-sm text-white/90 backdrop-blur-md">
        Phase 0 preview
      </span>
    </header>
  )
}
