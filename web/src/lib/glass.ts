/**
 * Moves the liquid-glass highlight to the pointer: on every pointer move over a glass
 * element, its --glass-x / --glass-y CSS variables are set to the pointer's position inside
 * it (see .glass in index.css). One listener on the document serves every glass element.
 */
const GLASS = '.glass, .glass-primary, .glass-tile'

export function followPointerHighlight(): void {
  document.addEventListener(
    'pointermove',
    (event) => {
      const target =
        event.target instanceof Element ? event.target.closest<HTMLElement>(GLASS) : null
      if (!target) return
      const box = target.getBoundingClientRect()
      target.style.setProperty('--glass-x', `${event.clientX - box.left}px`)
      target.style.setProperty('--glass-y', `${event.clientY - box.top}px`)
    },
    { passive: true },
  )
}
