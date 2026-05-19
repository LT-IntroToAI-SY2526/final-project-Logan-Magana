import streamlit as st
from PIL import Image
import io
import base64
from papercraft import generate_papercraft_pdf, generate_net_preview

st.set_page_config(
    page_title="PixelCraft — Papercraft Net Generator",
    page_icon="🎲",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=DM+Sans:wght@300;400;500&display=swap');

/* Page background */
[data-testid="stAppViewContainer"] {
    background: #0d0d14;
    background-image:
        radial-gradient(ellipse at 20% 10%, #1a0a2e 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, #0a1a2e 0%, transparent 50%);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Main container */
[data-testid="stMain"] {
    font-family: 'DM Sans', sans-serif;
}
[data-testid="block-container"] {
    padding-top: 2rem;
    max-width: 720px;
}

/* Hero title */
.hero-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 1.6rem;
    color: #fff;
    text-align: center;
    line-height: 1.6;
    margin-bottom: 0.25rem;
    text-shadow: 0 0 30px #a78bfa88, 0 0 60px #a78bfa44;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #8b8ba0;
    text-align: center;
    margin-bottom: 2.5rem;
    letter-spacing: 0.03em;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #16161f;
    border: 1.5px dashed #3d3d5c;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a78bfa;
}
[data-testid="stFileUploader"] label {
    color: #8b8ba0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.6rem !important;
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 20px #7c3aed55 !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55rem !important;
    background: linear-gradient(135deg, #059669, #34d399) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px #05966955 !important;
    letter-spacing: 0.05em !important;
}

/* Sliders */
[data-testid="stSlider"] label {
    color: #c4c4d4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Selectbox */
[data-testid="stSelectbox"] label {
    color: #c4c4d4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #16161f !important;
    border: 1px solid #3d3d5c !important;
    color: #e0e0f0 !important;
    border-radius: 8px !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: #c4c4d4 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Cards / info boxes */
.info-card {
    background: #16161f;
    border: 1px solid #2a2a3d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.info-card h4 {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.55rem;
    color: #a78bfa;
    margin: 0 0 0.75rem 0;
    letter-spacing: 0.08em;
}
.info-card p {
    color: #8b8ba0;
    font-size: 0.88rem;
    margin: 0.3rem 0;
    line-height: 1.6;
}

/* Step badges */
.step-badge {
    display: inline-block;
    font-family: 'Press Start 2P', monospace;
    font-size: 0.45rem;
    color: #a78bfa;
    border: 1px solid #3d3d5c;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    margin-bottom: 0.75rem;
    letter-spacing: 0.1em;
}

/* Color palette preview */
.color-swatch {
    display: inline-block;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    margin: 2px;
    border: 1px solid #ffffff22;
}

/* Image preview container */
.preview-wrap {
    background: #16161f;
    border: 1px solid #2a2a3d;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #2a2a3d;
    margin: 2rem 0;
}

/* Streamlit image */
[data-testid="stImage"] img {
    border-radius: 8px;
    image-rendering: pixelated;
}

/* Text colors */
p, li { color: #c4c4d4; font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { color: #e0e0f0; }

/* Number inputs */
[data-testid="stNumberInput"] label { color: #c4c4d4 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stNumberInput"] input { background: #16161f !important; color: #e0e0f0 !important; border: 1px solid #3d3d5c !important; border-radius: 8px !important; }

/* Column success message */
[data-testid="stAlert"] { border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">PIXEL<br>CRAFT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">turn any pixel art sprite into a printable papercraft cube</div>', unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("✦ how it works", expanded=False):
    st.markdown("""
    <div class="info-card">
        <h4>STEPS</h4>
        <p>① Upload your pixel art sprite (PNG with transparency works best)</p>
        <p>② Adjust the cube size and face options below</p>
        <p>③ Hit Generate — the app builds a flat net with your sprite on the front face</p>
        <p>④ Download the PDF, print it, cut along the solid lines, fold the dashed lines, and glue the tabs</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">STEP 01 — UPLOAD</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop your sprite here",
    type=["png", "gif", "jpg", "jpeg"],
    help="PNG with transparent background works best. Ideal size: 16×16 to 64×64 px."
)

if uploaded:
    img_bytes = uploaded.read()
    sprite = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = sprite.size

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
        st.image(sprite, caption=f"uploaded — {w}×{h}px", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Extract dominant colors
        from papercraft import get_dominant_colors
        colors = get_dominant_colors(sprite, n=6)
        swatches = "".join(
            f'<span class="color-swatch" style="background:rgb({r},{g},{b})" title="rgb({r},{g},{b})"></span>'
            for r, g, b, *_ in colors
        )
        st.markdown(f"""
        <div class="info-card">
            <h4>SPRITE INFO</h4>
            <p>Size: {w} × {h} px</p>
            <p>Mode: {sprite.mode}</p>
            <p>Dominant colors:</p>
            <p>{swatches}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Options ───────────────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 02 — CONFIGURE</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        face_size_cm = st.slider(
            "Face size (cm)",
            min_value=3, max_value=12, value=6, step=1,
            help="Each face of the cube will be this size when printed."
        )
        tab_size_cm = st.slider(
            "Glue tab size (cm)",
            min_value=1, max_value=3, value=1, step=1,
            help="Tabs used to glue the cube together."
        )

    with col_b:
        side_style = st.selectbox(
            "Side faces style",
            ["Dominant color", "Gradient fade", "Checkerboard pattern", "Blank (white)"],
            help="How the 5 non-front faces look."
        )
        add_instructions = st.checkbox("Include folding instructions page", value=True)

    st.markdown("---")

    # ── Generate ──────────────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 03 — GENERATE</div>', unsafe_allow_html=True)

    if st.button("⬡  Generate Papercraft Net"):
        with st.spinner("Building your cube net..."):
            try:
                # Generate preview image of the net
                preview_img = generate_net_preview(
                    sprite,
                    face_size_px=300,
                    side_style=side_style,
                )
                st.markdown("**Net preview** — this is what will be printed:")
                st.image(preview_img, use_container_width=True)

                # Generate PDF
                pdf_bytes = generate_papercraft_pdf(
                    sprite,
                    face_size_cm=face_size_cm,
                    tab_size_cm=tab_size_cm,
                    side_style=side_style,
                    add_instructions=add_instructions,
                )

                st.markdown("---")
                st.markdown('<div class="step-badge">STEP 04 — DOWNLOAD & PRINT</div>', unsafe_allow_html=True)
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
                    <p>• Cut along solid black lines</p>
                    <p>• Fold along dashed lines — score first for clean folds</p>
                    <p>• Use PVA glue or a glue stick on the tabs</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("Make sure your image is a valid PNG/GIF sprite.")

else:
    # Empty state illustration
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #3d3d5c;">
        <div style="font-size: 4rem; margin-bottom: 1rem; filter: grayscale(1) opacity(0.3);">🎲</div>
        <p style="font-family: 'DM Sans', sans-serif; color: #3d3d5c; font-size: 0.9rem;">
            upload a sprite above to get started
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style="text-align:center; font-size:0.75rem; color:#3d3d5c;">
    PixelCraft — pixel art papercraft generator
</p>
""", unsafe_allow_html=True)
