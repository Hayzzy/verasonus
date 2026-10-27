/**
 * Decorative background echoing the slides: the orange-to-pink "acrylic" panels of the title
 * slide and the soft corner glow. Pure CSS, so no image files or licences are involved.
 * On desktop the panels sit top-right; on narrower screens they move down behind the upload
 * card so the headline stays on black.
 */
export function Backdrop() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[72rem] overflow-hidden lg:h-[60rem]"
    >
      <div className="glow absolute -top-72 -right-56 size-[46rem] opacity-35 blur-2xl" />

      {/* Fluted orange panel */}
      <div
        className="absolute top-[30rem] right-[-10rem] h-[34rem] w-[20rem] rotate-[-16deg] rounded-[3rem] opacity-60 lg:-top-48 lg:right-[22%] lg:opacity-75"
        style={{
          background:
            'repeating-linear-gradient(90deg, rgb(255 255 255 / 0.12) 0 2px, transparent 2px 18px), linear-gradient(180deg, #ffb35c 0%, #ff7a00 48%, #d9481f 100%)',
        }}
      />

      {/* Frosted coral-to-pink panel */}
      <div
        className="absolute top-[36rem] right-[-24rem] h-[30rem] w-[36rem] rotate-[18deg] rounded-[3.5rem] opacity-55 sm:right-[-18rem] lg:top-24 lg:right-[-10rem] lg:opacity-75"
        style={{
          background:
            'linear-gradient(155deg, #fbaa56 0%, #f77256 32%, #f99682 52%, #d6779b 78%, #b45c8a 100%)',
          boxShadow: 'inset 0 1px 0 rgb(255 255 255 / 0.5), inset 0 0 0 1px rgb(255 255 255 / 0.18)',
        }}
      />

      {/* Fade the panels into the black page, keeping the text side dark */}
      <div className="absolute inset-0 bg-linear-to-b from-transparent via-ink/35 to-ink" />
      <div className="absolute inset-y-0 left-0 w-3/5 bg-linear-to-r from-ink via-ink/80 to-transparent" />
    </div>
  )
}
