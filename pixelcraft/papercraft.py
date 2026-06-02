"""
papercraft.py — Backend logic for PixelCraft.
Generates cube nets, PDFs, and 6-face orthographic renders from pixel art sprites.
"""

from PIL import Image, ImageDraw, ImageFont
import io
import math
from pathlib import Path
import numpy as np
from voxelizer import VoxelGrid, render_all_faces, smooth_grid


# ── PDF render DPI ────────────────────────────────────────────────────────────
# Print-quality: 300 DPI gives crisp edges at any face size ≥ 3 cm.
# (96 DPI was the old screen DPI and produced visibly blurry printed output.)
PDF_DPI = 300


# ── Color / background utilities ──────────────────────────────────────────────

def get_dominant_colors(img: Image.Image, n: int = 6):
    rgba = img.convert("RGBA")
    pixels = [px for px in rgba.getdata() if px[3] > 10]
    if not pixels:
        return [(200, 200, 200, 255)] * n
    from collections import Counter
    quantized = [((r >> 3) << 3, (g >> 3) << 3, (b >> 3) << 3, a) for r, g, b, a in pixels]
    top = [c for c, _ in Counter(quantized).most_common(n * 3)]
    filtered = [c for c in top if 10 < (c[0]+c[1]+c[2])/3 < 240][:n]
    while len(filtered) < n:
        filtered.append((150, 150, 180, 255))
    return filtered


def get_bg_color(img: Image.Image) -> tuple[int, int, int]:
    """
    Detect the background color robustly.

    Priority order:
      1. Transparency: if >4% of pixels are transparent, bg is a neutral
         light grey (the actual transparent areas are masked out by the sprite paste).
      2. Corner agreement: sample the 4 corners. If they agree within tol=30
         per channel, that IS the background.
      3. Fallback: dominant bright color.
    """
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    H, W = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    if (alpha < 20).sum() > H * W * 0.04:
        return (240, 240, 245)

    corners = np.array([
        rgba[0,   0,   :3],
        rgba[0,   W-1, :3],
        rgba[H-1, 0,   :3],
        rgba[H-1, W-1, :3],
    ], dtype=np.float32)
    corner_mean = corners.mean(axis=0)
    corner_std  = corners.std(axis=0).max()
    if corner_std < 30:
        return tuple(int(v) for v in corner_mean)

    r, g, b, *_ = get_dominant_colors(img, n=1)[0]
    return (r, g, b)


def make_checkerboard(size, color1, color2, tile=None):
    tile = tile or max(size // 8, 4)
    img = Image.new("RGB", (size, size), color1)
    draw = ImageDraw.Draw(img)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                draw.rectangle([x, y, x+tile-1, y+tile-1], fill=color2)
    return img


def make_gradient_face(size, color1, color2):
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / size
        draw.line([(0, y), (size, y)],
                  fill=tuple(int(color1[i]*(1-t)+color2[i]*t) for i in range(3)))
    return img


def prepare_front_face(sprite, size):
    face = Image.new("RGB", (size, size), get_bg_color(sprite))
    s = sprite.copy()
    scale = min(size / s.width, size / s.height)
    new_w = max(1, int(s.width  * scale))
    new_h = max(1, int(s.height * scale))
    s = s.resize((new_w, new_h), Image.NEAREST)
    ox, oy = (size - new_w) // 2, (size - new_h) // 2
    face.paste(s, (ox, oy), s if s.mode == "RGBA" else None)
    return face


def prepare_back_face(sprite, size, external_region_map=None):
    """
    Generate a clean back face:
      - Same silhouette derived from the sprite's occupancy mask (immune to
        the black-outline-on-black-bg rendering issue).
      - Interior filled with per-row dominant BRIGHT colour (dark/outline pixels
        excluded so eyes/beaks don't bleed through).
      - Horizontally mirrored (back-of-head = left/right flip).

    Vectorized: replaces the previous O(disp_h × disp_w) Python pixel loop
    with numpy broadcast operations for ~50× speedup at face_size=300.
    """
    from voxelizer import _resize, _bg_mask, DARK_THRESHOLD

    small   = _resize(sprite.convert("RGBA"))
    rgba_s  = np.array(small, dtype=np.uint8)
    is_bg_s = _bg_mask(rgba_s)
    occ_s   = ~is_bg_s          # True = sprite pixel
    cols_s  = rgba_s[:, :, :3]
    H_s, W_s = occ_s.shape

    bg_rgb = get_bg_color(sprite)

    # ── Per-row dominant bright colour ────────────────────────────────────────
    row_colors: list[tuple | None] = []
    for y in range(H_s):
        row_occ = occ_s[y, :]
        if not row_occ.any():
            row_colors.append(None)
            continue
        row_cols = cols_s[y, row_occ]
        bright   = row_cols[row_cols.max(axis=1) >= DARK_THRESHOLD]
        if len(bright) == 0:
            row_colors.append(None)
        else:
            row_colors.append(tuple(int(v) for v in np.median(bright, axis=0)))

    # Fill None rows: forward pass then backward pass using nearest real colour.
    last: tuple = bg_rgb
    for i in range(len(row_colors)):
        if row_colors[i] is not None:
            last = row_colors[i]
        else:
            row_colors[i] = last

    # Backward pass: fix bottom-only dark rows that the forward pass left as bg_rgb.
    last = bg_rgb
    for i in range(len(row_colors) - 1, -1, -1):
        # Use "not equal to the default fallback" as the signal, not "!= bg_rgb",
        # because bg_rgb itself can be a valid body colour on opaque-bg sprites.
        # Instead we track whether ANY real color was found after the first pass.
        if row_colors[i] is not None and row_colors[i] != bg_rgb:
            last = row_colors[i]
        elif last != bg_rgb and row_colors[i] == bg_rgb:
            row_colors[i] = last

    # ── Build face via numpy broadcast (replaces Python pixel loop) ───────────
    disp_scale = min(size / W_s, size / H_s)
    disp_w     = int(W_s * disp_scale)
    disp_h     = int(H_s * disp_scale)
    ox_px      = (size - disp_w) // 2
    oy_px      = (size - disp_h) // 2

    # Scale occupancy mask to display size.
    occ_img    = Image.fromarray((occ_s * 255).astype(np.uint8), "L")
    occ_scaled = np.array(occ_img.resize((disp_w, disp_h), Image.NEAREST)) > 128

    # Build a (disp_h, 3) array of per-row fill colors.
    row_color_arr = np.array(
        [row_colors[min(int(py / disp_h * H_s), H_s - 1)] or bg_rgb
         for py in range(disp_h)],
        dtype=np.uint8,
    )  # shape (disp_h, 3)

    # Broadcast to (disp_h, disp_w, 3): each row gets its fill color.
    fill_arr = np.broadcast_to(row_color_arr[:, None, :], (disp_h, disp_w, 3)).copy()

    # Start with full background, then stamp the sprite region.
    back_arr = np.full((size, size, 3), bg_rgb, dtype=np.uint8)
    # Where occupied → use fill color; where not occupied → keep bg.
    region = np.where(occ_scaled[:, :, None], fill_arr, np.array(bg_rgb, dtype=np.uint8))
    back_arr[oy_px:oy_px+disp_h, ox_px:ox_px+disp_w] = region

    # Mirror horizontally.
    back_arr = back_arr[:, ::-1, :]

    return Image.fromarray(back_arr, "RGB")


# ── Net layout ────────────────────────────────────────────────────────────────
NET_LAYOUT = {
    "top":    (1, 0),
    "left":   (0, 1),
    "front":  (1, 1),
    "right":  (2, 1),
    "back":   (3, 1),
    "bottom": (1, 2),
}


def _resolve_faces(sprite, size, side_style, use_3d_renders=True,
                   external_region_map=None):
    if use_3d_renders:
        grid  = VoxelGrid.build(sprite, external_region_map=external_region_map)
        grid  = smooth_grid(grid)
        bg    = get_bg_color(sprite)
        faces = render_all_faces(grid, face_size=size, bg=bg)
        faces["front"] = prepare_front_face(sprite, size)
        faces["back"]  = prepare_back_face(sprite, size, external_region_map)
        return faces
    return {
        "front":  prepare_front_face(sprite, size),
        "back":   prepare_back_face(sprite, size, external_region_map),
        "top":    _make_side(size, side_style, sprite),
        "bottom": _make_side(size, side_style, sprite),
        "left":   _make_side(size, side_style, sprite),
        "right":  _make_side(size, side_style, sprite),
    }


def _make_side(size, style, sprite):
    colors = get_dominant_colors(sprite, n=4)
    c1 = colors[0][:3]
    c2 = colors[1][:3] if len(colors) > 1 else (180, 180, 200)
    if style == "Dominant color":         return Image.new("RGB", (size, size), c1)
    elif style == "Gradient fade":        return make_gradient_face(size, c1, c2)
    elif style == "Checkerboard pattern": return make_checkerboard(size, c1, c2)
    else:                                 return Image.new("RGB", (size, size), (255, 255, 255))


# ── Net preview & PDF ─────────────────────────────────────────────────────────

def generate_net_preview(sprite, face_size_px=300, side_style="Dominant color",
                         use_3d_renders=True, external_region_map=None):
    fs    = face_size_px
    faces = _resolve_faces(sprite, fs, side_style, use_3d_renders, external_region_map)
    pad   = 20
    canvas = Image.new("RGB", (4*fs + 2*pad, 3*fs + 2*pad), (245, 245, 250))
    draw   = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for name, (col, row) in NET_LAYOUT.items():
        x, y = pad + col*fs, pad + row*fs
        canvas.paste(faces[name], (x, y))
        draw.rectangle([x, y, x+fs, y+fs], outline=(30, 30, 50), width=2)
        label = name.upper()
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx, ty = x + (fs-tw)//2, y + 8
        draw.rounded_rectangle([tx-6, ty-3, tx+tw+6, ty+th+3], radius=4,
                                fill=(100, 60, 200) if name == "front" else (0, 0, 0))
        draw.text((tx, ty), label, fill="white", font=font)
        _draw_fold_lines(draw, name, col, row, fs, pad)

    fc = NET_LAYOUT["front"]
    fx, fy = pad + fc[0]*fs, pad + fc[1]*fs
    draw.rectangle([fx, fy, fx+fs, fy+fs], outline=(120, 60, 255), width=3)
    return canvas


def _draw_fold_lines(draw, name, col, row, fs, pad):
    x, y     = pad + col*fs, pad + row*fs
    occupied = {v: k for k, v in NET_LAYOUT.items()}

    def dashed(x1, y1, x2, y2):
        dx, dy = x2-x1, y2-y1
        length = math.sqrt(dx*dx + dy*dy)
        steps  = int(length / 8)
        for i in range(steps):
            if i % 2 == 0:
                draw.line([(x1+dx*i/steps, y1+dy*i/steps),
                           (x1+dx*(i+1)/steps, y1+dy*(i+1)/steps)],
                          fill=(100, 100, 130), width=1)

    if (col, row-1) in occupied: dashed(x, y, x+fs, y)
    if (col, row+1) in occupied: dashed(x, y+fs, x+fs, y+fs)
    if (col-1, row) in occupied: dashed(x, y, x, y+fs)
    if (col+1, row) in occupied: dashed(x+fs, y, x+fs, y+fs)


def generate_papercraft_pdf(sprite, face_size_cm=6.0, tab_size_cm=1.0,
                            side_style="Dominant color", add_instructions=True,
                            use_3d_renders=True, external_region_map=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    fs, tab   = face_size_cm * cm, tab_size_cm * cm
    PAGE_W, PAGE_H = A4
    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=A4)
    c.setTitle("PixelCraft Papercraft Net")

    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.4, 0.2, 0.9)
    c.drawCentredString(PAGE_W/2, PAGE_H - 1.2*cm, "PIXELCRAFT — Cube Net")
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.5, 0.5, 0.6)
    source = "3D orthographic renders" if use_3d_renders else side_style
    region_note = " · custom regions" if external_region_map is not None else ""
    c.drawCentredString(PAGE_W/2, PAGE_H - 1.9*cm,
                        f"Face: {face_size_cm}cm  |  Tab: {tab_size_cm}cm  |  Faces: {source}{region_note}")

    # Use PDF_DPI (300) instead of 96 so faces are sharp when printed.
    # At 6 cm face: 300 DPI → 709 px vs old 96 DPI → 227 px.
    face_px   = int(face_size_cm / 2.54 * PDF_DPI)
    faces_img = _resolve_faces(sprite, face_px, side_style, use_3d_renders, external_region_map)

    net_w = 4*fs + 2*tab
    net_h = 3*fs + 2*tab
    ox    = (PAGE_W - net_w) / 2 + tab
    oy    = (PAGE_H - net_h) / 2 - 0.5*cm

    def face_origin(col, row):
        return ox + col*fs, oy + (2-row)*fs

    for face_name, (col, row) in NET_LAYOUT.items():
        x, y = face_origin(col, row)
        fbuf  = io.BytesIO()
        faces_img[face_name].save(fbuf, format="PNG")
        fbuf.seek(0)
        c.drawImage(ImageReader(fbuf), x, y, width=fs, height=fs)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColorRGB(*(0.5, 0.2, 1.0) if face_name == "front" else (0.3, 0.3, 0.4))
        c.drawCentredString(x + fs/2, y + fs - 14, face_name.upper())

    _draw_pdf_lines(c, ox, oy, fs, tab)
    _draw_pdf_tabs(c, ox, oy, fs, tab)

    ly = oy - 1.2*cm
    c.setFont("Helvetica", 8); c.setFillColorRGB(0.3, 0.3, 0.4)
    c.setStrokeColorRGB(0.1, 0.1, 0.15); c.setLineWidth(1.2); c.setDash()
    c.line(ox, ly, ox+1.2*cm, ly); c.drawString(ox+1.4*cm, ly-3, "Cut line")
    c.setStrokeColorRGB(0.4, 0.4, 0.6); c.setLineWidth(0.7); c.setDash(4, 4)
    c.line(ox+3.5*cm, ly, ox+4.7*cm, ly); c.setDash()
    c.drawString(ox+4.9*cm, ly-3, "Fold line")
    c.setFillColorRGB(0.85, 0.95, 0.85)
    c.rect(ox+7*cm, ly-5, 1*cm, 10, fill=1, stroke=0)
    c.setFillColorRGB(0.3, 0.3, 0.4); c.drawString(ox+8.2*cm, ly-3, "Glue tab")

    c.showPage()
    if add_instructions:
        _draw_instructions_page(c, PAGE_W, PAGE_H)
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


def _draw_pdf_lines(c, ox, oy, fs, tab):
    occupied = {v: k for k, v in NET_LAYOUT.items()}
    for _, (col, row) in NET_LAYOUT.items():
        x, y = ox + col*fs, oy + (2-row)*fs
        for direction, (x1,y1), (x2,y2) in [
            ("top",    (x, y+fs), (x+fs, y+fs)),
            ("bottom", (x, y),    (x+fs, y)),
            ("left",   (x, y),    (x, y+fs)),
            ("right",  (x+fs, y), (x+fs, y+fs)),
        ]:
            nc = {"top":(col,row-1),"bottom":(col,row+1),
                  "left":(col-1,row),"right":(col+1,row)}[direction]
            if nc in occupied:
                c.setStrokeColorRGB(0.4, 0.4, 0.6); c.setLineWidth(0.7); c.setDash(4, 4)
            else:
                c.setStrokeColorRGB(0.1, 0.1, 0.15); c.setLineWidth(1.2); c.setDash()
            c.line(x1, y1, x2, y2); c.setDash()


def _draw_pdf_tabs(c, ox, oy, fs, tab):
    occupied = {v: k for k, v in NET_LAYOUT.items()}
    c.setFillColorRGB(0.82, 0.94, 0.82)
    c.setStrokeColorRGB(0.1, 0.1, 0.15)
    c.setLineWidth(1.0); c.setDash()
    for _, (col, row) in NET_LAYOUT.items():
        x, y = ox + col*fs, oy + (2-row)*fs
        for direction, nc in [("top",(col,row-1)),("bottom",(col,row+1)),
                               ("left",(col-1,row)),("right",(col+1,row))]:
            if nc not in occupied:
                t = tab
                if direction == "top":
                    pts = [(x+t*.3,y+fs),(x+fs-t*.3,y+fs),(x+fs-t*.5,y+fs+t),(x+t*.5,y+fs+t)]
                elif direction == "bottom":
                    pts = [(x+t*.3,y),(x+fs-t*.3,y),(x+fs-t*.5,y-t),(x+t*.5,y-t)]
                elif direction == "left":
                    pts = [(x,y+t*.3),(x,y+fs-t*.3),(x-t,y+fs-t*.5),(x-t,y+t*.5)]
                else:
                    pts = [(x+fs,y+t*.3),(x+fs,y+fs-t*.3),(x+fs+t,y+fs-t*.5),(x+fs+t,y+t*.5)]
                path = c.beginPath()
                path.moveTo(*pts[0])
                for pt in pts[1:]: path.lineTo(*pt)
                path.close()
                c.drawPath(path, fill=1, stroke=1)


def _draw_instructions_page(c, PAGE_W, PAGE_H):
    from reportlab.lib.units import cm
    c.setFont("Helvetica-Bold", 18); c.setFillColorRGB(0.4, 0.2, 0.9)
    c.drawCentredString(PAGE_W/2, PAGE_H-2*cm, "How to assemble your cube")
    steps = [
        ("1. Print",  "Print at 100% scale on cardstock (200gsm). Do not scale to fit."),
        ("2. Score",  "Score all dashed fold lines with a ruler and empty ballpoint pen."),
        ("3. Cut",    "Cut along all solid black lines, including around the glue tabs."),
        ("4. Fold",   "Valley-fold all dashed lines inward. Fold tabs outward."),
        ("5. Glue",   "Apply glue to each tab. Assemble from bottom upward. Hold 30s per join."),
        ("6. Done!",  "Your pixel art cube is complete!"),
    ]
    y = PAGE_H - 3.5*cm
    for title, body in steps:
        c.setFont("Helvetica-Bold", 12); c.setFillColorRGB(0.4, 0.2, 0.9)
        c.drawString(2.5*cm, y, title); y -= 0.55*cm
        c.setFont("Helvetica", 10); c.setFillColorRGB(0.2, 0.2, 0.3)
        line = ""
        for word in body.split():
            test = (line + " " + word).strip()
            if len(test) > 80:
                c.drawString(2.5*cm, y, line); y -= 0.5*cm; line = word
            else:
                line = test
        if line:
            c.drawString(2.5*cm, y, line); y -= 0.8*cm
    c.setFillColorRGB(0.93, 0.90, 1.0)
    c.roundRect(2*cm, y-1.5*cm, PAGE_W-4*cm, 2.2*cm, 8, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 10); c.setFillColorRGB(0.4, 0.2, 0.9)
    c.drawString(2.5*cm, y-0.6*cm, "Pro tip:")
    c.setFont("Helvetica", 10); c.setFillColorRGB(0.2, 0.2, 0.3)
    c.drawString(2.5*cm, y-1.15*cm,
                 "Use a craft knife + cutting mat. Laminate for a long-lasting finish.")
    c.setFont("Helvetica", 8); c.setFillColorRGB(0.6, 0.6, 0.7)
    c.drawCentredString(PAGE_W/2, 1.5*cm, "Made with PixelCraft")
