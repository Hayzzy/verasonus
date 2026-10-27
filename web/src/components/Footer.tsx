import { Logo } from './Logo'

export function Footer() {
  return (
    <footer className="border-t border-white/10">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 pt-8 pb-[max(2rem,env(safe-area-inset-bottom))] text-sm text-mist sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <p className="flex items-center gap-2.5">
          <Logo className="size-6" />
          <span>
            <span className="font-display font-semibold text-white">VeraSonus</span> · Latin for
            “true sound”
          </span>
        </p>
        <p>NUS-ISS Pattern Recognition Systems · Shrey Yadav &amp; Sushant Maramkulam Shiji</p>
      </div>
    </footer>
  )
}
