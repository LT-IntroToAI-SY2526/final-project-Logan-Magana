"""
papercraft.py
Core logic for PixelCraft — generates cube nets and PDFs from pixel art sprites.
"""

from PIL import Image, ImageDraw, ImageFont
import io
import math


# ── Color utilities ───────────────────────────────────────────────────────────

def get_dominant_colors(img: Image.Image, n: int = 6):
    """Return the n most common non-transparent colors in the image."""
    rgba = img.convert("RGBA")
    pixels = [
        px for px in rgba.getdata()
        if px[3] > 10  # skip near-transparent
    ]
    if not pixels:
        return [(200, 200, 200, 255)] * n

    # Count colors (quantize slightly to reduce noise)
    from collections import Counter
    quantized = [
        ((r >> 3) << 3, (g >> 3) << 3, (b >> 3) << 3, a)
        for r, g, b, a in pixels
    ]
    counts = Counter(quantized)
    top = [color for color, _ in counts.most_common(n * 3)]

    # Filter out near-black and near-white background artifacts
    filtered = []
    for c in top:
        r, g, b, a = c
        brightness = (r + g + b) / 3
        if brightness < 240 and brightness > 10:
            filtered.append(c)
        if len(filtered) >= n:
            break

    # Pad if needed
    while len(filtered) < n:
        filtered.append((150, 150, 180, 255))

    return filtered[:n]


def get_bg_color(img: Image.Image):
    """Pick a solid background color from the sprite's dominant palette."""
    colors = get_dominant_colors(img, n=3)
    r, g, b, *_ = colors[0]
    return (r, g, b)


def make_checkerboard(size: int, color1, color2, tile: int = None):
    """Create a checkerboard pattern image."""
    if tile is None:
        tile = max(size // 8, 4)
    img = Image.new("RGB", (size, size), color1)
    draw = ImageDraw.Draw(img)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=color2)
    return img


def make_gradient_face(size: int, color1, color2):
    """Vertical gradient from color1 to color2."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / size
        r = int(color1[0] * (1 - t) + color2[0] * t)
        g = int(color1[1] * (1 - t) + color2[1] * t)
        b = int(color1[2] * (1 - t) + color2[2] * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    return img


def make_side_face(size: int, style: str, sprite: Image.Image):
    """Generate a side face based on the chosen style."""
    colors = get_dominant_colors(sprite, n=4)
    c1 = colors[0][:3]
    c2 = colors[1][:3] if len(colors) > 1 else (180, 180, 200)

    if style == "Dominant color":
        return Image.new("RGB", (size, size), c1)
    elif style == "Gradient fade":
        return make_gradient_face(size, c1, c2)
    elif style == "Checkerboard pattern":
        return make_checkerboard(size, c1, c2)
    else:  # Blank
        return Image.new("RGB", (size, size), (255, 255, 255))


def prepare_front_face(sprite: Image.Image, size: int) -> Image.Image:
    """Scale sprite to face size, compositing onto dominant background color."""
    bg_color = get_bg_color(sprite)
    face = Image.new("RGB", (size, size), bg_color)

    # Scale sprite to fit, preserving aspect ratio, pixel-perfect
    s = sprite.copy()
    s.thumbnail((size, size), Image.NEAREST)
    sw, sh = s.size

    # Center it
    ox = (size - sw) // 2
    oy = (size - sh) // 2

    if s.mode == "RGBA":
        face.paste(s, (ox, oy), s)
    else:
        face.paste(s, (ox, oy))

    return face


# ── Net layout ────────────────────────────────────────────────────────────────
# Classic cross-shaped cube net:
#
#         [TOP]
#  [LEFT] [FRONT] [RIGHT] [BACK]
#         [BOTTOM]
#
# Positions in (col, row) grid units:
NET_LAYOUT = {
    "top":    (1, 0),
    "left":   (0, 1),
    "front":  (1, 1),
    "right":  (2, 1),
    "back":   (3, 1),
    "bottom": (1, 2),
}


def generate_net_preview(
    sprite: Image.Image,
    face_size_px: int = 300,
    side_style: str = "Dominant color",
    tab_size_px: int = 0,
) -> Image.Image:
    """
    Build a flat preview image of the cube net.
    Returns a PIL Image.
    """
    fs = face_size_px

    # Build face images
    faces = {
        "front": prepare_front_face(sprite, fs),
        "top":    make_side_face(fs, side_style, sprite),
        "bottom": make_side_face(fs, side_style, sprite),
        "left":   make_side_face(fs, side_style, sprite),
        "right":  make_side_face(fs, side_style, sprite),
        "back":   make_side_face(fs, side_style, sprite),
    }

    # Canvas: 4 faces wide, 3 faces tall + padding
    pad = 20
    canvas_w = 4 * fs + 2 * pad
    canvas_h = 3 * fs + 2 * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 250))
    draw = ImageDraw.Draw(canvas)

    # Paste faces
    for name, (col, row) in NET_LAYOUT.items():
        x = pad + col * fs
        y = pad + row * fs
        canvas.paste(faces[name], (x, y))

        # Draw solid border
        draw.rectangle([x, y, x + fs, y + fs], outline=(30, 30, 50), width=2)

        # Label
        label = name.upper()
        # Simple centered label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x + (fs - tw) // 2
        ty = y + 8

        # Background pill for label
        draw.rounded_rectangle(
            [tx - 6, ty - 3, tx + tw + 6, ty + th + 3],
            radius=4,
            fill=(0, 0, 0, 180) if name != "front" else (100, 60, 200),
        )
        draw.text((tx, ty), label, fill="white", font=font)

        # Draw fold/cut indicators (dashed lines between faces)
        _draw_fold_lines(draw, name, col, row, fs, pad)

    # Add a "FRONT ★" special marker
    fc = NET_LAYOUT["front"]
    fx = pad + fc[0] * fs
    fy = pad + fc[1] * fs
    draw.rectangle([fx, fy, fx + fs, fy + fs], outline=(120, 60, 255), width=3)

    return canvas


def _draw_fold_lines(draw, name, col, row, fs, pad):
    """Draw dashed fold lines along shared edges."""
    x = pad + col * fs
    y = pad + row * fs

    def dashed_line(x1, y1, x2, y2, color=(100, 100, 130), dash=8):
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return
        steps = int(length / dash)
        for i in range(steps):
            if i % 2 == 0:
                t0 = i / steps
                t1 = (i + 1) / steps
                draw.line(
                    [(x1 + dx * t0, y1 + dy * t0), (x1 + dx * t1, y1 + dy * t1)],
                    fill=color, width=1,
                )

    # Shared edges become fold lines (dashed)
    neighbors = {
        "top":    [("bottom", "front")],
        "left":   [("right", "front")],
        "front":  [("right", "right"), ("top", "top"), ("bottom", "bottom"), ("left", "left")],
        "right":  [("left", "front")],
        "back":   [("left", "right")],
        "bottom": [("top", "front")],
    }

    # Simply draw dashed lines on all 4 edges of each face
    # Outer edges = cut lines (solid, already drawn as border)
    # We draw inner (shared) edges as dashed
    all_cols = {v[0] for v in NET_LAYOUT.values()}
    all_rows = {v[1] for v in NET_LAYOUT.values()}

    # Check neighbors
    occupied = {v: k for k, v in NET_LAYOUT.items()}

    # Top edge shared?
    if (col, row - 1) in occupied:
        dashed_line(x, y, x + fs, y)
    # Bottom edge shared?
    if (col, row + 1) in occupied:
        dashed_line(x, y + fs, x + fs, y + fs)
    # Left edge shared?
    if (col - 1, row) in occupied:
        dashed_line(x, y, x, y + fs)
    # Right edge shared?
    if (col + 1, row) in occupied:
        dashed_line(x + fs, y, x + fs, y + fs)


# ── PDF generation ────────────────────────────────────────────────────────────

def generate_papercraft_pdf(
    sprite: Image.Image,
    face_size_cm: float = 6.0,
    tab_size_cm: float = 1.0,
    side_style: str = "Dominant color",
    add_instructions: bool = True,
) -> bytes:
    """
    Generate a printable PDF with the cube net.
    Returns raw PDF bytes.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    # Convert sizes
    fs = face_size_cm * cm
    tab = tab_size_cm * cm

    PAGE_W, PAGE_H = A4
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # ── Page 1: The Net ───────────────────────────────────────────────────────
    c.setTitle("PixelCraft Papercraft Net")

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.4, 0.2, 0.9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.2 * cm, "PIXELCRAFT — Cube Net")
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.5, 0.5, 0.6)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.9 * cm, f"Face size: {face_size_cm}cm  |  Tab: {tab_size_cm}cm  |  Sides: {side_style}")

    # Render faces at print resolution (96px per cm → face_size_cm * 96)
    px_per_cm = 96
    face_px = int(face_size_cm * px_per_cm)

    faces_img = {
        "front":  prepare_front_face(sprite, face_px),
        "top":    make_side_face(face_px, side_style, sprite),
        "bottom": make_side_face(face_px, side_style, sprite),
        "left":   make_side_face(face_px, side_style, sprite),
        "right":  make_side_face(face_px, side_style, sprite),
        "back":   make_side_face(face_px, side_style, sprite),
    }

    # Net origin — center the 4-wide × 3-tall net on the page
    net_w = 4 * fs + 2 * tab  # with side tabs
    net_h = 3 * fs + 2 * tab
    origin_x = (PAGE_W - net_w) / 2 + tab
    origin_y = (PAGE_H - net_h) / 2 - 0.5 * cm  # slight downward offset for header

    def face_origin(col, row):
        """Bottom-left corner of face in PDF coordinates (reportlab is bottom-up)."""
        x = origin_x + col * fs
        # PDF y=0 is bottom; convert row from top-down to bottom-up
        y = origin_y + (2 - row) * fs
        return x, y

    # Draw each face
    for face_name, (col, row) in NET_LAYOUT.items():
        x, y = face_origin(col, row)
        pil_img = faces_img[face_name]

        # Convert PIL to reportlab ImageReader
        face_buf = io.BytesIO()
        pil_img.save(face_buf, format="PNG")
        face_buf.seek(0)
        rl_img = ImageReader(face_buf)

        c.drawImage(rl_img, x, y, width=fs, height=fs)

        # Label overlay
        c.setFont("Helvetica-Bold", 7)
        if face_name == "front":
            c.setFillColorRGB(0.5, 0.2, 1.0)
        else:
            c.setFillColorRGB(0.3, 0.3, 0.4)
        c.drawCentredString(x + fs / 2, y + fs - 14, face_name.upper())

    # Draw cut lines (solid) and fold lines (dashed) in PDF
    _draw_pdf_lines(c, origin_x, origin_y, fs, tab)

    # Draw glue tabs
    _draw_pdf_tabs(c, origin_x, origin_y, fs, tab)

    # Legend
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.3, 0.3, 0.4)
    legend_y = origin_y - 1.2 * cm

    # Cut line sample
    c.setStrokeColorRGB(0.1, 0.1, 0.15)
    c.setLineWidth(1.2)
    c.line(origin_x, legend_y, origin_x + 1.2 * cm, legend_y)
    c.drawString(origin_x + 1.4 * cm, legend_y - 3, "Cut line")

    # Fold line sample
    c.setStrokeColorRGB(0.4, 0.4, 0.6)
    c.setLineWidth(0.7)
    c.setDash(4, 4)
    c.line(origin_x + 3.5 * cm, legend_y, origin_x + 4.7 * cm, legend_y)
    c.setDash()
    c.drawString(origin_x + 4.9 * cm, legend_y - 3, "Fold line")

    # Tab sample
    c.setFillColorRGB(0.85, 0.95, 0.85)
    c.rect(origin_x + 7 * cm, legend_y - 5, 1 * cm, 10, fill=1, stroke=0)
    c.setFillColorRGB(0.3, 0.3, 0.4)
    c.drawString(origin_x + 8.2 * cm, legend_y - 3, "Glue tab")

    c.showPage()

    # ── Page 2: Instructions ──────────────────────────────────────────────────
    if add_instructions:
        _draw_instructions_page(c, PAGE_W, PAGE_H)
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()


def _draw_pdf_lines(c, ox, oy, fs, tab):
    """Draw cut (solid) and fold (dashed) lines on the net."""
    from reportlab.lib.units import cm

    occupied = {v: k for k, v in NET_LAYOUT.items()}

    for face_name, (col, row) in NET_LAYOUT.items():
        x = ox + col * fs
        # PDF coords: bottom-up
        y = oy + (2 - row) * fs

        edges = [
            ("top",    (x, y + fs), (x + fs, y + fs)),
            ("bottom", (x, y),      (x + fs, y)),
            ("left",   (x, y),      (x, y + fs)),
            ("right",  (x + fs, y), (x + fs, y + fs)),
        ]

        for direction, (x1, y1), (x2, y2) in edges:
            dc = {"top": (col, row - 1), "bottom": (col, row + 1),
                  "left": (col - 1, row), "right": (col + 1, row)}[direction]

            if dc in occupied:
                # Shared edge → fold line
                c.setStrokeColorRGB(0.4, 0.4, 0.6)
                c.setLineWidth(0.7)
                c.setDash(4, 4)
            else:
                # Outer edge → cut line
                c.setStrokeColorRGB(0.1, 0.1, 0.15)
                c.setLineWidth(1.2)
                c.setDash()

            c.line(x1, y1, x2, y2)
            c.setDash()


def _draw_pdf_tabs(c, ox, oy, fs, tab):
    """Draw glue tabs on outer edges of the net."""
    from reportlab.lib.units import cm

    occupied = {v: k for k, v in NET_LAYOUT.items()}

    c.setFillColorRGB(0.82, 0.94, 0.82)
    c.setStrokeColorRGB(0.1, 0.1, 0.15)
    c.setLineWidth(1.0)
    c.setDash()

    tab_faces = [
        # (face, edge direction, tab polygon points offset)
        ("top",    "top"),
        ("left",   "left"),
        ("right",  "right"),
        ("back",   "right"),
        ("bottom", "bottom"),
        ("front",  "bottom"),  # only draw tab on unshared edges
    ]

    for face_name, (col, row) in NET_LAYOUT.items():
        x = ox + col * fs
        y = oy + (2 - row) * fs

        edges = {
            "top":    ((col, row - 1), [(x, y + fs), (x + fs, y + fs), (x + fs + 0, y + fs + tab), (x, y + fs + tab)]),
            "bottom": ((col, row + 1), [(x, y), (x + fs, y), (x + fs, y - tab), (x, y - tab)]),
            "left":   ((col - 1, row), [(x, y), (x, y + fs), (x - tab, y + fs), (x - tab, y)]),
            "right":  ((col + 1, row), [(x + fs, y), (x + fs, y + fs), (x + fs + tab, y + fs), (x + fs + tab, y)]),
        }

        for direction, (neighbor, pts) in edges.items():
            if neighbor not in occupied:
                # Outer edge — draw tab
                # Taper the tab slightly
                if direction == "top":
                    pts = [(x + tab * 0.3, y + fs), (x + fs - tab * 0.3, y + fs),
                           (x + fs - tab * 0.5, y + fs + tab), (x + tab * 0.5, y + fs + tab)]
                elif direction == "bottom":
                    pts = [(x + tab * 0.3, y), (x + fs - tab * 0.3, y),
                           (x + fs - tab * 0.5, y - tab), (x + tab * 0.5, y - tab)]
                elif direction == "left":
                    pts = [(x, y + tab * 0.3), (x, y + fs - tab * 0.3),
                           (x - tab, y + fs - tab * 0.5), (x - tab, y + tab * 0.5)]
                elif direction == "right":
                    pts = [(x + fs, y + tab * 0.3), (x + fs, y + fs - tab * 0.3),
                           (x + fs + tab, y + fs - tab * 0.5), (x + fs + tab, y + tab * 0.5)]

                path = c.beginPath()
                path.moveTo(*pts[0])
                for pt in pts[1:]:
                    path.lineTo(*pt)
                path.close()
                c.drawPath(path, fill=1, stroke=1)


def _draw_instructions_page(c, PAGE_W, PAGE_H):
    """Draw a simple folding instructions page."""
    from reportlab.lib.units import cm

    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0.4, 0.2, 0.9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 2 * cm, "How to assemble your cube")

    steps = [
        ("1. Print", "Print this sheet at 100% scale on cardstock (200gsm recommended). Do not scale to fit."),
        ("2. Score",  "Use a ruler and an empty ballpoint pen to score along all dashed fold lines. This makes folding clean and crisp."),
        ("3. Cut",    "Cut along all solid black lines, including around the glue tabs. Take your time — clean cuts make better cubes."),
        ("4. Fold",   "Fold all dashed lines inward (valley folds). Fold tabs outward so they will be hidden inside the cube."),
        ("5. Glue",   "Apply glue or double-sided tape to each tab. Assemble the cube starting from the bottom, working upward. Hold each join for 30 seconds."),
        ("6. Done!",  "Your pixel art cube is complete. Display it, photograph it, or make a whole collection!"),
    ]

    y = PAGE_H - 3.5 * cm
    for title, body in steps:
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.4, 0.2, 0.9)
        c.drawString(2.5 * cm, y, title)
        y -= 0.55 * cm
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.2, 0.2, 0.3)

        # Simple word-wrap
        words = body.split()
        line = ""
        for word in words:
            test = (line + " " + word).strip()
            if len(test) > 80:
                c.drawString(2.5 * cm, y, line)
                y -= 0.5 * cm
                line = word
            else:
                line = test
        if line:
            c.drawString(2.5 * cm, y, line)
            y -= 0.8 * cm

    # Tip box
    c.setFillColorRGB(0.93, 0.90, 1.0)
    c.roundRect(2 * cm, y - 1.5 * cm, PAGE_W - 4 * cm, 2.2 * cm, 8, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.4, 0.2, 0.9)
    c.drawString(2.5 * cm, y - 0.6 * cm, "Pro tip:")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.2, 0.2, 0.3)
    c.drawString(2.5 * cm, y - 1.15 * cm,
                 "Laminate or spray with matte varnish for a long-lasting finish. "
                 "Use a craft knife + cutting mat for best results.")

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.6, 0.6, 0.7)
    c.drawCentredString(PAGE_W / 2, 1.5 * cm, "Made with PixelCraft")
