import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import io
import json
from pathlib import Path
from papercraft import (
    generate_papercraft_pdf,
    generate_net_preview,
    get_dominant_colors,
    get_bg_color,
)
from voxelizer import VoxelGrid, render_all_faces, smooth_grid, export_grid_json

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


def load_viewer(grid_data: dict, sprite_w: int, sprite_h: int) -> str:
    """Load viewer.html and inject voxelizer grid JSON + sprite dims."""
    viewer_path = Path(__file__).parent / "viewer.html"
    html = viewer_path.read_text(encoding="utf-8")
    html = html.replace("__GRID_DATA__", json.dumps(grid_data))
    html = html.replace("__SPRITE_W__",  str(sprite_w))
    html = html.replace("__SPRITE_H__",  str(sprite_h))
    return html


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">PIXEL<br>CRAFT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">turn any pixel art sprite into a printable papercraft cube</div>', unsafe_allow_html=True)

with st.expander("✦ how it works", expanded=False):
    st.markdown("""
    <div class="info-card">
        <h4>STEPS</h4>
        <p>① Upload your pixel art sprite (PNG with transparency works best)</p>
        <p>② Preview the 3D voxel model — rotate, zoom, switch voxel shapes</p>
        <p>③ Configure cube size and generate — all 6 faces are auto-rendered from each angle</p>
        <p>④ Download the PDF, print, cut, fold, and glue</p>
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
    sprite    = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h      = sprite.size

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

    # ── Step 2: 3D Viewer (preview only) ─────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 02 — 3D PREVIEW</div>', unsafe_allow_html=True)
    st.markdown("Explore your sprite as a 3D voxel model — rotate, zoom, switch shapes:")

    preview_sprite = sprite.copy()
    MAX_DIM = 64
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        preview_sprite = preview_sprite.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST
        )
        st.caption(f"⚡ Downscaled to {preview_sprite.width}×{preview_sprite.height} for 3D performance.")

    with st.spinner("Building 3D model…"):
        grid = VoxelGrid.build(preview_sprite)
        grid = smooth_grid(grid)
        grid_data = export_grid_json(grid)
    # Check occupancy via depthMap (voxel_count key doesn't exist in export_grid_json)
    has_voxels = any(grid_data['depthMap'][y][x] > 0
                     for y in range(grid_data['H'])
                     for x in range(grid_data['W']))
    if not has_voxels:
        st.warning("No visible pixels found — check your image has non-transparent content.")
    else:
        viewer_html = load_viewer(grid_data, preview_sprite.width, preview_sprite.height)
        components.html(viewer_html, height=460, scrolling=False)

    st.markdown("---")

    # ── Step 3: Configure ─────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 03 — CONFIGURE</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        face_size_cm     = st.slider("Face size (cm)", 3, 12, 6, 1)
        tab_size_cm      = st.slider("Glue tab size (cm)", 1, 3, 1, 1)
    with col_b:
        use_3d_renders   = st.checkbox(
            "Auto-render all 6 faces from 3D model angles",
            value=True,
            help="Each face of the cube shows the correct orthographic view of your sprite."
        )
        side_style       = st.selectbox(
            "Fallback style (if 3D renders off)",
            ["Dominant color", "Gradient fade", "Checkerboard pattern", "Blank (white)"],
        )
        add_instructions = st.checkbox("Include folding instructions page", value=True)

    # Live face preview
    if use_3d_renders:
        with st.expander("👁 Preview auto-rendered faces", expanded=False):
            with st.spinner("Rendering 6 faces…"):
                _g = VoxelGrid.build(sprite)
                _g = smooth_grid(_g)
                faces_preview = render_all_faces(_g, face_size=120, bg=get_bg_color(sprite))
            face_order = ["front", "back", "left", "right", "top", "bottom"]
            cols = st.columns(6)
            for i, name in enumerate(face_order):
                with cols[i]:
                    st.image(faces_preview[name], caption=name, use_container_width=True)

    st.markdown("---")

    # ── Step 4: Generate ──────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 04 — GENERATE</div>', unsafe_allow_html=True)

    if st.button("⬡  Generate Papercraft Net"):
        with st.spinner("Building your cube net…"):
            try:
                preview_img = generate_net_preview(
                    sprite, face_size_px=300,
                    side_style=side_style,
                    use_3d_renders=use_3d_renders,
                )
                st.markdown("**Net preview:**")
                st.image(preview_img, use_container_width=True)

                pdf_bytes = generate_papercraft_pdf(
                    sprite,
                    face_size_cm=face_size_cm,
                    tab_size_cm=tab_size_cm,
                    side_style=side_style,
                    add_instructions=add_instructions,
                    use_3d_renders=use_3d_renders,
                )

                st.markdown("---")
                st.markdown('<div class="step-badge">STEP 05 — DOWNLOAD & PRINT</div>', unsafe_allow_html=True)
                st.success("Your papercraft net is ready!")
                st.download_button(
                    label="⬇  Download PDF",
                    data=pdf_bytes,
                    file_name="pixelcraft_net.pdf",
                    mime="application/pdf",
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
