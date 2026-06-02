import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from papercraft import (
    generate_papercraft_pdf,
    generate_net_preview,
    get_dominant_colors,
    get_bg_color,
)
from voxelizer import (
    VoxelGrid,
    render_all_faces,
    smooth_grid,
    export_grid_json,
    classify_regions,
    REGION_HEAD, REGION_TORSO, REGION_LIMB, REGION_TUFT,
)

st.set_page_config(
    page_title="PixelCraft — Papercraft Net Generator",
    page_icon="🎲",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=DM+Sans:wght@300;400;500&display=swap');
[data-testid="stAppViewContainer"] {
    background: #0d0d14;
    background-image: radial-gradient(ellipse at 20% 10%, #1a0a2e 0%, transparent 50%),
                      radial-gradient(ellipse at 80% 90%, #0a1a2e 0%, transparent 50%);
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stMain"] { font-family: 'DM Sans', sans-serif; }
[data-testid="block-container"] { padding-top: 2rem; max-width: 720px; }
.hero-title {
    font-family: 'Press Start 2P', monospace; font-size: 1.6rem; color: #fff;
    text-align: center; line-height: 1.6; margin-bottom: 0.25rem;
    text-shadow: 0 0 30px #a78bfa88, 0 0 60px #a78bfa44;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif; font-size: 1rem; color: #8b8ba0;
    text-align: center; margin-bottom: 2.5rem; letter-spacing: 0.03em;
}
[data-testid="stFileUploader"] {
    background: #16161f; border: 1.5px dashed #3d3d5c;
    border-radius: 12px; padding: 1rem; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #a78bfa; }
[data-testid="stFileUploader"] label { color: #8b8ba0 !important; }
.stButton > button {
    font-family: 'Press Start 2P', monospace !important; font-size: 0.6rem !important;
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important; width: 100% !important;
    box-shadow: 0 4px 20px #7c3aed55 !important; letter-spacing: 0.05em !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
[data-testid="stDownloadButton"] > button {
    font-family: 'Press Start 2P', monospace !important; font-size: 0.55rem !important;
    background: linear-gradient(135deg, #059669, #34d399) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important; width: 100% !important;
    box-shadow: 0 4px 20px #05966955 !important; letter-spacing: 0.05em !important;
}
[data-testid="stSlider"] label, [data-testid="stSelectbox"] label,
[data-testid="stCheckbox"] label { color: #c4c4d4 !important; }
[data-testid="stSelectbox"] > div > div {
    background: #16161f !important; border: 1px solid #3d3d5c !important;
    color: #e0e0f0 !important; border-radius: 8px !important;
}
.info-card {
    background: #16161f; border: 1px solid #2a2a3d;
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1rem 0;
}
.info-card h4 {
    font-family: 'Press Start 2P', monospace; font-size: 0.55rem;
    color: #a78bfa; margin: 0 0 0.75rem 0; letter-spacing: 0.08em;
}
.info-card p { color: #8b8ba0; font-size: 0.88rem; margin: 0.3rem 0; line-height: 1.6; }
.step-badge {
    display: inline-block; font-family: 'Press Start 2P', monospace;
    font-size: 0.45rem; color: #a78bfa; border: 1px solid #3d3d5c;
    border-radius: 20px; padding: 0.3rem 0.8rem; margin-bottom: 0.75rem; letter-spacing: 0.1em;
}
.color-swatch { display: inline-block; width: 24px; height: 24px; border-radius: 4px; margin: 2px; border: 1px solid #ffffff22; }
.preview-wrap { background: #16161f; border: 1px solid #2a2a3d; border-radius: 12px; padding: 1rem; text-align: center; }
hr { border: none; border-top: 1px solid #2a2a3d; margin: 2rem 0; }
[data-testid="stImage"] img { border-radius: 8px; image-rendering: pixelated; }
p, li { color: #c4c4d4; font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { color: #e0e0f0; }
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sprite_hash(img_bytes: bytes) -> str:
    """SHA-256 of full file content — collision-proof unlike hash(bytes[:64])."""
    return hashlib.sha256(img_bytes).hexdigest()


def sprite_to_b64(sprite: Image.Image) -> str:
    buf = io.BytesIO()
    sprite.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def _cached_auto_region_map_json(sprite_hash: str, img_bytes: bytes) -> str:
    """
    Compute auto-region JSON once per unique sprite and cache it.
    The sprite_hash key ensures the cache invalidates when the image changes.
    img_bytes is passed so the function is pure (hash alone could theoretically
    collide across Streamlit sessions, though SHA-256 makes this negligible).
    """
    from voxelizer import _resize, _bg_mask
    sprite = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    small  = _resize(sprite)
    rgba   = np.array(small, dtype=np.uint8)
    is_bg  = _bg_mask(rgba)
    occ    = ~is_bg
    rmap   = classify_regions(occ)
    rows   = []
    for y in range(rmap.shape[0]):
        row = []
        for x in range(rmap.shape[1]):
            row.append(int(rmap[y, x]) if occ[y, x] else None)
        rows.append(row)
    return json.dumps(rows)


@st.cache_data(show_spinner=False)
def _cached_painter_template() -> str:
    """Read region_painter_final.html once and cache it for the process lifetime."""
    return (Path(__file__).parent / "region_painter_final.html").read_text(encoding="utf-8")


def write_painter_static(sprite: Image.Image, img_bytes: bytes, return_url: str) -> None:
    """
    Inject sprite data into the cached painter template and write to ./static/painter.html.
    Template reading is cached; only the string replacements + disk write happen per call.
    """
    from voxelizer import _resize
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    small  = _resize(sprite.convert("RGBA"))
    b64    = sprite_to_b64(small)
    auto_j = _cached_auto_region_map_json(_sprite_hash(img_bytes), img_bytes)

    html = _cached_painter_template()
    html = html.replace("__SPRITE_B64__",   b64)
    html = html.replace("__AUTO_REGIONS__", auto_j)
    html = html.replace("__SPRITE_W__",     str(small.width))
    html = html.replace("__SPRITE_H__",     str(small.height))
    html = html.replace("__RETURN_URL__",   return_url)

    (static_dir / "painter.html").write_text(html, encoding="utf-8")


def load_viewer(grid_data: dict, sprite_w: int, sprite_h: int) -> str:
    html = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")
    html = html.replace("__GRID_DATA__", json.dumps(grid_data))
    html = html.replace("__SPRITE_W__",  str(sprite_w))
    html = html.replace("__SPRITE_H__",  str(sprite_h))
    return html


def decode_regions_b64(b64_str: str):
    """Decode base64 region map from painter. Returns (ndarray, w, h) or None."""
    try:
        raw = base64.b64decode(b64_str)
        if len(raw) < 4:
            return None
        w = (raw[0] << 8) | raw[1]
        h = (raw[2] << 8) | raw[3]
        if len(raw) < 4 + w * h or w <= 0 or h <= 0:
            return None
        data = np.frombuffer(raw[4:4 + w * h], dtype=np.uint8).astype(np.int8) - 1
        return data.reshape(h, w), w, h
    except Exception:
        return None


# ── Decode incoming region map from URL query params ─────────────────────────
_qp_regions = st.query_params.get("regions", None)
if _qp_regions and "external_region_map" not in st.session_state:
    decoded = decode_regions_b64(_qp_regions)
    if decoded is not None:
        rmap, rw, rh = decoded
        st.session_state["external_region_map"] = rmap
        st.session_state["region_map_source"]   = "custom"

if "external_region_map" not in st.session_state:
    st.session_state["external_region_map"] = None
if "region_map_source"   not in st.session_state:
    st.session_state["region_map_source"]   = "auto"
# Cache the built VoxelGrid between Step 3 (viewer) and Step 5 (generate)
# so we don't build it twice for the same sprite + settings.
if "cached_grid"         not in st.session_state:
    st.session_state["cached_grid"]         = None
if "cached_grid_key"     not in st.session_state:
    st.session_state["cached_grid_key"]     = None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">PIXEL<br>CRAFT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">turn any pixel art sprite into a printable papercraft cube</div>', unsafe_allow_html=True)

with st.expander("✦ how it works", expanded=False):
    st.markdown("""
    <div class="info-card">
        <h4>STEPS</h4>
        <p>① Upload your pixel art sprite (PNG with transparency works best)</p>
        <p>② (Optional) Open the region painter to control 3D depth per body part</p>
        <p>③ Preview the 3D voxel model — rotate, zoom, switch voxel shapes</p>
        <p>④ Configure cube size and generate — all 6 faces auto-rendered</p>
        <p>⑤ Download the PDF, print, cut, fold, and glue</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">STEP 01 — UPLOAD SPRITE</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop your sprite here",
    type=["png", "gif", "jpg", "jpeg"],
    help="PNG with transparent background works best. Ideal: 16×16 to 64×64 px."
)

if uploaded:
    img_bytes = uploaded.read()

    # SHA-256 hash: collision-proof, uses full content not just first 64 bytes.
    sprite_id = _sprite_hash(img_bytes)
    if st.session_state.get("last_sprite_id") != sprite_id:
        st.session_state["external_region_map"] = None
        st.session_state["region_map_source"]   = "auto"
        st.session_state["last_sprite_id"]      = sprite_id
        st.session_state["cached_grid"]         = None
        st.session_state["cached_grid_key"]     = None
        st.query_params.clear()

    sprite = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h   = sprite.size

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
        st.image(sprite, caption=f"uploaded — {w}×{h}px", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        colors   = get_dominant_colors(sprite, n=6)
        swatches = "".join(
            f'<span class="color-swatch" style="background:rgb({r},{g},{b})"></span>'
            for r, g, b, *_ in colors
        )
        st.markdown(f"""
        <div class="info-card">
            <h4>SPRITE INFO</h4>
            <p>Size: {w} × {h} px</p>
            <p>Mode: {sprite.mode}</p>
            <p>Dominant colors: {swatches}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 2: Region Painter ────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 02 — REGION PAINTER (optional)</div>', unsafe_allow_html=True)

    if st.session_state["region_map_source"] == "custom":
        st.success("✦ Using **custom regions** from painter — driving the 3D model below.")
        if st.button("⟳  Reset to auto-detect", key="reset_regions"):
            st.session_state["external_region_map"] = None
            st.session_state["region_map_source"]   = "auto"
            st.session_state["cached_grid"]         = None
            st.session_state["cached_grid_key"]     = None
            st.query_params.clear()
            st.rerun()
    else:
        st.info("ℹ Auto-detect active. Open the painter to manually assign regions.")

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx      = get_script_run_ctx()
        port     = getattr(ctx, "server_port", 8501)
        base_url = f"http://localhost:{port}"
    except Exception:
        base_url = "http://localhost:8501"

    write_painter_static(sprite, img_bytes, return_url=base_url)
    painter_url = base_url + "/app/static/painter.html"

    st.markdown(f"""
    <div style="margin:0.5rem 0 0.75rem 0;">
      <a href="{painter_url}" target="_blank"
         style="display:inline-block; padding:10px 22px;
                background:linear-gradient(135deg,#7c3aed,#a78bfa);
                color:white; border-radius:9px; font-weight:700; font-size:0.85rem;
                text-decoration:none; letter-spacing:0.03em;
                box-shadow:0 4px 16px #7c3aed44;">
        🎨 Open Region Painter ↗
      </a>
      <span style="color:#6b6b8a; font-size:0.8rem; margin-left:12px;">
        Opens in a new tab — click <strong>Apply regions →</strong> when done.
      </span>
    </div>
    <div style="background:#16161f;border:1px solid #2a2a3d;border-radius:10px;padding:0.9rem 1rem;margin-bottom:0.5rem;">
      <p style="color:#8b8ba0;font-size:0.83rem;margin:0;line-height:1.8;">
        <strong style="color:#f97316;">HEAD</strong> — deepest &nbsp;·&nbsp;
        <strong style="color:#3b82f6;">TORSO</strong> — medium &nbsp;·&nbsp;
        <strong style="color:#22c55e;">LIMBS</strong> — shallow &nbsp;·&nbsp;
        <strong style="color:#a855f7;">TUFT</strong> — shallowest (hair, antennae)<br>
        <span style="color:#5a5a7a;">Unpainted pixels fall back to auto-detect.</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["external_region_map"] is not None:
        rm = st.session_state["external_region_map"]
        st.caption(
            f"🟠 Head: {int((rm==0).sum())} px  ·  "
            f"🔵 Torso: {int((rm==1).sum())} px  ·  "
            f"🟢 Limbs: {int((rm==2).sum())} px  ·  "
            f"🟣 Tuft: {int((rm==3).sum())} px"
        )

    st.markdown("---")

    # ── Step 3: 3D Viewer ─────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 03 — 3D PREVIEW</div>', unsafe_allow_html=True)
    st.markdown("Your sprite as a 3D voxel model — rotate, zoom, switch shapes:")

    preview_sprite = sprite.copy()
    MAX_DIM = 64
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        preview_sprite = preview_sprite.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST
        )
        st.caption(f"⚡ Downscaled to {preview_sprite.width}×{preview_sprite.height} for 3D performance.")

    # Cache key: sprite identity + region map source so the grid is reused
    # between the viewer render and the PDF generate step when nothing changed.
    ext_rm     = st.session_state["external_region_map"]
    grid_key   = (sprite_id, st.session_state["region_map_source"],
                  id(ext_rm) if ext_rm is not None else None)

    if st.session_state["cached_grid_key"] != grid_key:
        with st.spinner("Building 3D model…"):
            grid = VoxelGrid.build(preview_sprite, external_region_map=ext_rm)
            grid = smooth_grid(grid)
        st.session_state["cached_grid"]     = grid
        st.session_state["cached_grid_key"] = grid_key
    else:
        grid = st.session_state["cached_grid"]

    grid_data = export_grid_json(grid)

    has_voxels = any(grid_data['depthMap'][y][x] > 0
                     for y in range(grid_data['H'])
                     for x in range(grid_data['W']))
    if not has_voxels:
        st.warning("No visible pixels found — check your image has non-transparent content.")
    else:
        components.html(load_viewer(grid_data, preview_sprite.width, preview_sprite.height),
                        height=460, scrolling=False)

    st.markdown("---")

    # ── Step 4: Configure ─────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 04 — CONFIGURE</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        face_size_cm = st.slider("Face size (cm)", 3, 12, 6, 1)
        tab_size_cm  = st.slider("Glue tab size (cm)", 1, 3, 1, 1)
    with col_b:
        use_3d_renders = st.checkbox(
            "Auto-render all 6 faces from 3D model angles", value=True,
            help="Each face shows the correct orthographic view of your sprite."
        )
        side_style = st.selectbox(
            "Fallback style (if 3D renders off)",
            ["Dominant color", "Gradient fade", "Checkerboard pattern", "Blank (white)"],
        )
        add_instructions = st.checkbox("Include folding instructions page", value=True)

    if use_3d_renders:
        with st.expander("👁 Preview auto-rendered faces", expanded=False):
            with st.spinner("Rendering 6 faces…"):
                _g = VoxelGrid.build(sprite, external_region_map=ext_rm)
                _g = smooth_grid(_g)
                faces_preview = render_all_faces(_g, face_size=120, bg=get_bg_color(sprite))
            cols = st.columns(6)
            for i, name in enumerate(["front","back","left","right","top","bottom"]):
                with cols[i]:
                    st.image(faces_preview[name], caption=name, use_container_width=True)

    st.markdown("---")

    # ── Step 5: Generate ──────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 05 — GENERATE</div>', unsafe_allow_html=True)
    region_note = "custom regions" if st.session_state["region_map_source"] == "custom" else "auto-detected regions"
    st.caption(f"Will use {region_note} for depth generation.")

    if st.button("⬡  Generate Papercraft Net"):
        with st.spinner("Building your cube net…"):
            try:
                preview_img = generate_net_preview(
                    sprite, face_size_px=300, side_style=side_style,
                    use_3d_renders=use_3d_renders,
                    external_region_map=ext_rm,
                )
                st.markdown("**Net preview:**")
                st.image(preview_img, use_container_width=True)

                pdf_bytes = generate_papercraft_pdf(
                    sprite, face_size_cm=face_size_cm, tab_size_cm=tab_size_cm,
                    side_style=side_style, add_instructions=add_instructions,
                    use_3d_renders=use_3d_renders,
                    external_region_map=ext_rm,
                )

                st.markdown("---")
                st.markdown('<div class="step-badge">STEP 06 — DOWNLOAD & PRINT</div>', unsafe_allow_html=True)
                st.success("Your papercraft net is ready!")
                st.download_button(
                    label="⬇  Download PDF", data=pdf_bytes,
                    file_name="pixelcraft_net.pdf", mime="application/pdf",
                )
                st.markdown("""
                <div class="info-card">
                    <h4>PRINTING TIPS</h4>
                    <p>• Print at 100% scale — do not fit to page</p>
                    <p>• Use cardstock (200gsm+) for a sturdier cube</p>
                    <p>• Cut solid lines · Fold dashed lines · Glue the tabs</p>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                raise e

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;">
        <div style="font-size:4rem;margin-bottom:1rem;filter:grayscale(1) opacity(0.3);">🎲</div>
        <p style="color:#3d3d5c;font-size:0.9rem;">upload a sprite above to get started</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;font-size:0.75rem;color:#3d3d5c;">PixelCraft — pixel art papercraft generator</p>', unsafe_allow_html=True)
