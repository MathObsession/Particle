"""Render ParticleDesign green artwork using only Pillow and NumPy.

Run: python3 render.py --help
The mask preserves the letter shapes from the supplied Particle One image.
All color, light, texture, curves, and material effects are generated in code.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Code-only ParticleDesign brand artwork")
content = parser.add_mutually_exclusive_group()
content.add_argument("--mask", type=Path, help="Grayscale alpha mask, white foreground on black")
content.add_argument("--text", help="Custom wordmark text; defaults to the bundled Particle One mask")
parser.add_argument("--font", type=Path, help="Font file for --text")
parser.add_argument("--output", type=Path, default=Path.cwd() / "output")
parser.add_argument("--name", default="particle-design")
parser.add_argument("--width", type=int, default=3000)
parser.add_argument("--height", type=int, default=1384)
parser.add_argument("--circles", type=float, default=1.0, help="Circle intensity multiplier; 0 hides them")
parser.add_argument("--shine", type=float, default=0.7, help="Edge-shine multiplier; default is subtle")
parser.add_argument("--grain", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--background-only", action="store_true")
args = parser.parse_args()
if min(args.width, args.height) < 256 or max(args.width, args.height) > 6000:
    parser.error("Dimensions must be between 256 and 6000 pixels")
if any(value < 0 or value > 3 for value in (args.circles, args.shine, args.grain)):
    parser.error("Circles, shine, and grain must be between 0 and 3")
if Path(args.name).name != args.name or args.name in ("", ".", ".."):
    parser.error("--name must be a filename stem, not a path")
W, H = args.width, args.height
OUT = args.output.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
scale = W / 3000
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
u, v = x / W, y / H


def light(cx, cy, sx, sy):
    return np.exp(-0.5 * (((u - cx) / sx) ** 2 + ((v - cy) / sy) ** 2))


def overlay(image, color, mask, opacity=1.0):
    if opacity != 1:
        mask = mask.point(lambda q: int(q * opacity))
    return Image.composite(Image.new("RGB", image.size, color), image, mask)


# Irregular, overlapping color clouds reproduce the reference's curved green
# canopy and cool blue-mint corners without using a straight gradient ramp.
field = np.empty((H, W, 3), dtype=np.float32)
field[:] = [184, 220, 204]
warped_u = u + 0.055 * np.sin(v * 5.0 + u * 2.0)
warped_v = v + 0.10 * np.sin(u * 5.2 - 0.8)


def wash(color, cx, cy, sx, sy, strength=1.0, tilt=0.0):
    global field
    dx, dy = warped_u - cx, warped_v - cy
    dx, dy = dx + tilt * dy, dy - tilt * dx
    alpha = strength * np.exp(-0.5 * ((dx / sx) ** 2 + (dy / sy) ** 2))
    alpha = np.clip(alpha, 0, 1)[..., None]
    field = field * (1 - alpha) + np.array(color, dtype=np.float32) * alpha


wash([177, 213, 225], 0.64, 1.18, 0.65, 0.41, 0.94, -0.20)
wash([170, 222, 184], 1.08, 0.50, 0.32, 0.46, 0.75, 0.35)
wash([159, 217, 71], 0.48, -0.10, 0.37, 0.48, 0.98, -0.12)
wash([181, 226, 100], 0.02, 0.12, 0.32, 0.31, 0.68, 0.35)
wash([146, 208, 93], 0.81, 0.08, 0.21, 0.30, 0.40, -0.30)
wash([203, 232, 174], 0.42, 0.59, 0.30, 0.25, 0.46, -0.35)
wash([188, 231, 218], -0.08, 0.81, 0.31, 0.22, 0.65, 0.20)
wash([170, 212, 224], 0.01, 0.76, 0.25, 0.31, 0.48, -0.25)
wash([177, 214, 226], 0.94, 1.02, 0.30, 0.34, 0.58, 0.30)
wash([185, 231, 139], 0.51, 0.38, 0.22, 0.37, 0.53, -0.35)
wash([212, 237, 152], 0.10, 0.07, 0.21, 0.20, 0.30, 0.20)

# Faint large-scale pigment variations keep the washes tactile and organic.
mottle = Image.fromarray(np.uint8(rng.uniform(0, 255, (44, 95))))
mottle = mottle.resize((W, H), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(18))
field += (np.asarray(mottle, dtype=np.float32)[..., None] - 127.5) * 0.022
canvas = Image.fromarray(np.uint8(np.clip(field, 0, 255)))


# Oversized translucent curves flow through the corners like the reference.
lines = Image.new("L", (W, H))
pen = ImageDraw.Draw(lines)
for cx, cy, rx, ry, rotation, alpha in [
    (-0.16, 0.49, 0.49, 0.64, -0.30, 37),
    (1.16, 0.48, 0.49, 0.64, 0.30, 37),
    (0.50, -0.73, 0.79, 0.91, 0.0, 18),
    (0.50, 1.75, 0.79, 0.91, 0.0, 18),
]:
    angle = np.linspace(0, 2 * np.pi, 2000)
    ex, ey = rx * W * np.cos(angle), ry * H * np.sin(angle)
    px = cx * W + ex * np.cos(rotation) - ey * np.sin(rotation)
    py = cy * H + ex * np.sin(rotation) + ey * np.cos(rotation)
    pen.line(list(zip(px, py)), fill=alpha, width=12)
edge_fade = np.clip(0.12 + 0.88 * (np.exp(-(u / 0.25) ** 2) + np.exp(-((1 - u) / 0.25) ** 2)), 0, 1)
lines = Image.fromarray(np.uint8(np.clip(np.asarray(lines, dtype=np.float32) * edge_fade * args.circles, 0, 255)))
canvas = overlay(canvas, (243, 253, 225), lines.filter(ImageFilter.GaussianBlur(15)), 0.35)
canvas = overlay(canvas, (245, 254, 232), lines.filter(ImageFilter.GaussianBlur(5)))


background_only_canvas = canvas.copy()

# A larger mask and gentle bevel give the existing wordmark a pearly surface.
if args.text:
    candidates = [args.font] if args.font else [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    font_path = next((p for p in candidates if p and p.is_file()), None)
    if font_path is None:
        parser.error("Provide an installed font file with --font")
    font = ImageFont.truetype(str(font_path), 400)
    bounds = font.getbbox(args.text)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        parser.error("Text must contain visible characters")
    mask = Image.new("L", (bounds[2]-bounds[0]+16, bounds[3]-bounds[1]+16))
    ImageDraw.Draw(mask).text((8-bounds[0], 8-bounds[1]), args.text, font=font, fill=255)
else:
    mask = Image.open(args.mask or ROOT.parent / "Sources/ParticleOne/Resources/Art/wordmark-mask.png").convert("L")
if mask.getbbox() is None:
    parser.error("The mask contains no visible foreground")
mask = mask.crop(mask.getbbox())
logo_w = min(round(W * 0.725), round(H * 0.44 * mask.width / mask.height))
logo_h = max(1, round(logo_w * mask.height / mask.width))
mask = mask.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
full = Image.new("L", (W, H))
left, upper = (W - logo_w) // 2, (H - logo_h) // 2
full.paste(mask, (left, upper))
shadow = ImageChops.offset(full, 0, 13).filter(ImageFilter.GaussianBlur(22))
canvas = overlay(canvas, (95, 159, 70), shadow, 0.12)
glow = full.filter(ImageFilter.GaussianBlur(45))
canvas = overlay(canvas, (249, 255, 232), glow, 0.30)
close_glow = full.filter(ImageFilter.GaussianBlur(10))
canvas = overlay(canvas, (253, 255, 243), close_glow, 0.28)

# A luminous outer rim catches light immediately outside each letter.
outer_rim = ImageChops.subtract(full.filter(ImageFilter.MaxFilter(13)), full)
canvas = overlay(canvas, (255, 255, 247), outer_rim.filter(ImageFilter.GaussianBlur(8)), min(1, 0.42 * args.shine))
canvas = overlay(canvas, (255, 255, 251), outer_rim.filter(ImageFilter.GaussianBlur(2)), min(1, 0.28 * args.shine))

logo_v = np.clip((y - upper) / logo_h, 0, 1)
surface_top = np.array([255, 255, 249], dtype=np.float32)
surface_bottom = np.array([236, 250, 216], dtype=np.float32)
surface = surface_top[None, None, :] * (1 - logo_v[..., None])
surface += surface_bottom[None, None, :] * logo_v[..., None]
surface += light(0.39, 0.48, 0.30, 0.16)[..., None] * [2, 2, 3]
material = Image.fromarray(np.uint8(np.clip(surface, 0, 255)))
canvas = Image.composite(material, canvas, full.point(lambda q: int(q * 0.94)))

inner = full.filter(ImageFilter.GaussianBlur(2.6))
lower_edge = ImageChops.subtract(full, ImageChops.offset(inner, 0, -6))
upper_edge = ImageChops.subtract(full, ImageChops.offset(inner, 0, 5))
canvas = overlay(canvas, (217, 239, 192), lower_edge, 0.30)
canvas = overlay(canvas, (255, 255, 255), upper_edge, min(1, 0.70 * args.shine))

# A narrow pearl-white edge and soft directional highlight add restrained shine.
inner_rim = ImageChops.subtract(full, full.filter(ImageFilter.MinFilter(13)))
canvas = overlay(canvas, (255, 255, 255), inner_rim.filter(ImageFilter.GaussianBlur(0.8)), min(1, 0.60 * args.shine))
shine_field = np.clip(0.34 + 0.66 * light(0.38, 0.35, 0.42, 0.23), 0, 1)
shine = Image.fromarray(np.uint8(np.asarray(upper_edge, dtype=np.float32) * shine_field))
canvas = overlay(canvas, (255, 255, 255), shine.filter(ImageFilter.GaussianBlur(2)), min(1, 0.85 * args.shine))


# Fine monochromatic grain has a faint coarser layer, like printed gradients.
pixels = np.asarray(background_only_canvas if args.background_only else canvas, dtype=np.float32)
fine = rng.normal(0, 1.6, (H, W)).astype(np.float32)
coarse = Image.fromarray(np.uint8(rng.uniform(0, 255, (H // 2, W // 2))))
coarse = coarse.resize((W, H), Image.Resampling.BILINEAR)
grain = fine + (np.asarray(coarse, dtype=np.float32) - 127.5) * 0.067
pixels += grain[..., None] * args.grain
canvas = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))

# Rounded corners match the card shape in the reference. PNG is transparent
# outside the card; JPG uses a white page beneath it.
corners = Image.new("L", (W * 2, H * 2))
ImageDraw.Draw(corners).rounded_rectangle((0, 0, W * 2, H * 2), radius=round(96 * scale), fill=255)
corners = corners.resize((W, H), Image.Resampling.LANCZOS)
png = canvas.convert("RGBA")
png.putalpha(corners)
png.save(OUT / f"{args.name}.png", dpi=(300, 300))
jpg = Image.new("RGB", (W, H), "white")
jpg.paste(canvas, (0, 0), corners)
jpg.save(OUT / f"{args.name}.jpg", quality=97, subsampling=0, dpi=(300, 300))
preview = png.copy()
preview.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
preview.save(OUT / f"{args.name}-preview.png")
print(f"Created PNG and JPG: {W} x {H}")
