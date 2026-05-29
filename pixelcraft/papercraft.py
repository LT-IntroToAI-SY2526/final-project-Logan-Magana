"""
papercraft.py — Backend logic for PixelCraft.
Generates cube nets, PDFs, and 6-face orthographic renders from pixel art sprites.
No HTML/JS here — that lives in viewer.html.
"""

from PIL import Image, ImageDraw, ImageFont
import io
import math
from pathlib import Path
from voxelizer import VoxelGrid, render_all_faces, smooth_grid


# ── Color utilities ───────────────────────────────────────────────────────────

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


def get_bg_color(img: Image.Image):
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
        draw.line([(0, y), (size, y)], fill=tuple(int(color1[i]*(1-t)+color2[i]*t) for i in range(3)))
    return img


def prepare_front_face(sprite, size):
    face = Image.new("RGB", (size, size), get_bg_color(sprite))
    s = sprite.copy()
    s.thumbnail((size, size), Image.NEAREST)
    ox, oy = (size - s.width) // 2, (size - s.height) // 2
    face.paste(s, (ox, oy), s if s.mode == "RGBA" else None)
    return face


# ── Net layout ────────────────────────────────────────────────────────────────
NET_LAYOUT = {
    "top":    (1, 0),
    "left":   (0, 1),
    "front":  (1, 1),
    "right":  (2, 1),
    "back":   (3, 1),
    "bottom": (1, 2),
}


def _resolve_faces(sprite, size, side_style, use_3d_renders=True):
    if use_3d_renders:
        grid  = VoxelGrid.build(sprite)
        grid  = smooth_grid(grid)
        bg    = get_bg_color(sprite)
        faces = render_all_faces(grid, face_size=size, bg=bg)
        # Override front/back with original sprite so black line art is preserved
        # on those faces. Side/top/bottom come from the de-outlined voxel model.
        faces["front"] = prepare_front_face(sprite, size)
        faces["back"]  = prepare_front_face(
            sprite.transpose(Image.FLIP_LEFT_RIGHT), size)
        return faces
    return {
        "front":  prepare_front_face(sprite, size),
        "top":    _make_side(size, side_style, sprite),
        "bottom": _make_side(size, side_style, sprite),
        "left":   _make_side(size, side_style, sprite),
        "right":  _make_side(size, side_style, sprite),
        "back":   _make_side(size, side_style, sprite),
    }


def _make_side(size, style, sprite):
    colors = get_dominant_colors(sprite, n=4)
    c1, c2 = colors[0][:3], colors[1][:3] if len(colors) > 1 else (180, 180, 200)
    if style == "Dominant color":         return Image.new("RGB", (size, size), c1)
    elif style == "Gradient fade":        return make_gradient_face(size, c1, c2)
    elif style == "Checkerboard pattern": return make_checkerboard(size, c1, c2)
    else:                                 return Image.new("RGB", (size, size), (255, 255, 255))


# ── Net preview & PDF ─────────────────────────────────────────────────────────

def generate_net_preview(sprite, face_size_px=300, side_style="Dominant color",
                         use_3d_renders=True):
    fs    = face_size_px
    faces = _resolve_faces(sprite, fs, side_style, use_3d_renders)
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
                            use_3d_renders=True):
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
    c.drawCentredString(PAGE_W/2, PAGE_H - 1.9*cm,
                        f"Face: {face_size_cm}cm  |  Tab: {tab_size_cm}cm  |  Faces: {source}")

    face_px   = int(face_size_cm * 96)
    faces_img = _resolve_faces(sprite, face_px, side_style, use_3d_renders)

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
