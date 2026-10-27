"""Draw the VeraSonus logo as PNG/ICO icons for the web app (PWA) and the Chrome extension.

The geometry matches web/public/favicon.svg on a 100-unit grid: an orange gradient ring
(the slides' circle motif) around five white waveform bars. Re-run after changing the logo:

    python scripts/make_icons.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FLAME, EMBER, BLACK = (255, 122, 0), (213, 106, 70), (0, 0, 0, 255)
RING_RADIUS, RING_WIDTH, BAR_WIDTH = 34, 4.5, 4.5
BARS = [(33, 14), (41.5, 28), (50, 40), (58.5, 24), (67, 12)]  # (x centre, height)
SUPERSAMPLE = 4  # draw large, then shrink, for smooth edges


def draw_logo(size: int, corner_radius: float = 0) -> Image.Image:
    """Logo on black; ``corner_radius`` (grid units) rounds the background square."""
    s = size * SUPERSAMPLE
    u = s / 100  # pixels per grid unit

    background = Image.new("L", (s, s), 0)
    ImageDraw.Draw(background).rounded_rectangle([0, 0, s - 1, s - 1], corner_radius * u, 255)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (s, s), BLACK), mask=background)

    # Ring: a diagonal flame -> ember gradient, masked to an annulus.
    t = (np.add.outer(np.arange(s), np.arange(s)) / (2 * (s - 1)))[..., None]
    gradient = (np.array(FLAME) * (1 - t) + np.array(EMBER) * t).astype(np.uint8)
    ring = Image.new("L", (s, s), 0)
    r = (RING_RADIUS + RING_WIDTH / 2) * u
    ImageDraw.Draw(ring).ellipse(
        [50 * u - r, 50 * u - r, 50 * u + r, 50 * u + r], outline=255, width=round(RING_WIDTH * u)
    )
    img.paste(Image.fromarray(gradient, "RGB"), mask=ring)

    bars = ImageDraw.Draw(img)
    for x, h in BARS:
        box = [(x - BAR_WIDTH / 2) * u, (50 - h / 2) * u, (x + BAR_WIDTH / 2) * u, (50 + h / 2) * u]
        bars.rounded_rectangle(box, radius=BAR_WIDTH / 2 * u, fill=(255, 255, 255, 255))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    web = ROOT / "web" / "public"
    (web / "icons").mkdir(parents=True, exist_ok=True)
    # Full-bleed squares: iOS and Android apply their own masks. The ring sits inside the
    # maskable safe zone (a centred circle of 40% radius), so one design serves both.
    for name, size in [
        ("icon-192.png", 192),
        ("icon-512.png", 512),
        ("icon-maskable-512.png", 512),
        ("apple-touch-icon.png", 180),
    ]:
        draw_logo(size).save(web / "icons" / name)
    draw_logo(64, corner_radius=22).save(web / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])

    ext = ROOT / "extension" / "icons"
    ext.mkdir(parents=True, exist_ok=True)
    for size in (16, 48, 128):
        draw_logo(size, corner_radius=22).save(ext / f"icon-{size}.png")
    print("Icons written to web/public and extension/icons")


if __name__ == "__main__":
    main()
