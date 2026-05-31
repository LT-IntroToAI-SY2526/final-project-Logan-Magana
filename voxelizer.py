"""
voxelizer.py — Full voxelization pipeline for pixel-art sprites.

Design philosophy (chibi / Lego-style sculpt)
─────────────────────────────────────────────
• Outline / dark pixels are VISUAL LINE ART only — they inherit their depth
  from the nearest bright interior pixel via distance transform.  They never
  contribute structural depth data of their own.
• Depth contrast is deliberately low.  The face is a smooth rounded mass,
  not a cratered landscape.  Facial features sit ON the surface.
• Region depths are shallow and close together (HEAD=7, TORSO=7, LIMB=4).
  Chibi / Lego proportions mean the head is as massive as the torso.
• Elliptical profile minimum is 55 % so the silhouette reads as a rounded
  cylinder, not a pointed cone.
• Gaussian smoothing (sigma 1.2, 2 passes) blurs all remaining depth
  transitions into gentle slopes.
• Color blending (dark → nearest interior) is applied BROADLY before
  voxel colors are stored, so ALL views (side, top, bottom) show bright
  surface colors — no black shells anywhere.
"""

from __future__ import annotations
import math
import numpy as np
from PIL import Image

# ── tunables ──────────────────────────────────────────────────────────────────
MAX_DIM = 64

REGION_HEAD  = 0
REGION_TORSO = 1
REGION_LIMB  = 2
REGION_TUFT  = 3   # narrow top protrusion: feathers, hair tufts, antennae

# Chibi proportions: head ≈ torso mass, limbs thin, tufts shallow
REGION_MAX_DEPTH: dict[int, int] = {
    REGION_HEAD:  16,
    REGION_TORSO: 14,
    REGION_LIMB:  8,
    REGION_TUFT:  5,   # thin protruding accent — minimal depth
}
MAX_DEPTH      = 16   # deeper extrusion — side profile has real sculptural mass
DARK_THRESHOLD = 55   # max(r,g,b) below this → line art (for depth AND contour detection)
MIN_PROFILE    = 0.62 # ellipse floor — thick rounded cylinder, not a cone


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Pre-process
# ─────────────────────────────────────────────────────────────────────────────

def _resize(sprite: Image.Image) -> Image.Image:
    w, h = sprite.size
    if max(w, h) > MAX_DIM:
        s = MAX_DIM / max(w, h)
        sprite = sprite.resize((max(1, int(w*s)), max(1, int(h*s))), Image.NEAREST)
    return sprite


def _bg_mask(rgba: np.ndarray, tol: int = 30) -> np.ndarray:
    H, W   = rgba.shape[:2]
    alpha  = rgba[:, :, 3]
    if (alpha < 20).sum() > H * W * 0.04:
        return alpha < 20
    corners = [rgba[0,0,:3], rgba[0,W-1,:3], rgba[H-1,0,:3], rgba[H-1,W-1,:3]]
    bg_col  = np.mean(corners, axis=0).astype(np.float32)
    diff    = np.abs(rgba[:,:,:3].astype(np.float32) - bg_col[None,None,:])
    return diff.max(axis=2) < tol


def preprocess(sprite: Image.Image):
    sprite   = _resize(sprite)
    rgba     = np.array(sprite.convert("RGBA"), dtype=np.uint8)
    is_bg    = _bg_mask(rgba)
    occupied = ~is_bg
    colors   = rgba[:, :, :3].copy()
    return rgba, occupied, colors


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Region classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_regions(occupied: np.ndarray) -> np.ndarray:
    H, W        = occupied.shape
    region_map  = np.full((H, W), REGION_LIMB, dtype=np.int8)
    row_counts  = occupied.sum(axis=1)
    max_count   = max(int(row_counts.max()), 1)
    is_seam     = row_counts <= max(1, max_count * 0.15)

    segments = []
    i = 0
    while i < H:
        if not is_seam[i]:
            j = i
            while j < H and not is_seam[j]:
                j += 1
            segments.append((i, j, float(row_counts[i:j].mean())))
            i = j
        else:
            i += 1

    if not segments:
        return region_map

    widest = max(s[2] for s in segments)
    for k, (start, end, mean_w) in enumerate(segments):
        if   mean_w >= widest * 0.75: code = REGION_TORSO
        elif k == 0:                  code = REGION_HEAD
        else:                         code = REGION_LIMB
        region_map[start:end, :] = code

    first = segments[0]
    if first[2] >= widest * 0.75:
        he = first[0] + max(1, (first[1] - first[0]) // 4)
        region_map[first[0]:he, :] = REGION_HEAD

    # Detect narrow top protrusions (tufts, feathers, antennae).
    first_occ = next((y for y in range(H) if row_counts[y] > 0), None)
    if first_occ is not None:
        for y in range(first_occ, H):
            if row_counts[y] < widest * 0.40:
                region_map[y, :] = REGION_TUFT
            else:
                break

    return region_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Outline + dark pixel detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outlines(occupied: np.ndarray) -> np.ndarray:
    """Pixels that touch the background (or the image edge) in 4-connectivity."""
    H, W       = occupied.shape
    is_outline = np.zeros((H, W), dtype=bool)
    padded     = np.pad(occupied, 1, constant_values=False)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        nb = padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        is_outline |= (occupied & ~nb)
    return is_outline


def detect_dark_pixels(colors: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Max-channel brightness below DARK_THRESHOLD → line art, not structure."""
    brightness = colors[:, :, :3].max(axis=2)
    return occupied & (brightness.astype(np.int32) < DARK_THRESHOLD)


def _detect_contour_darks(colors: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """
    Identify dark pixels that are LINE ART CONTOURS vs intentional FEATURES.

    CONTOUR (replace color → blend toward skin):
      • Any dark component that touches the background  — silhouette outline
      • Any dark component whose bounding box is only 1px thick in either
        dimension — interior border stripes (belt, collar, region dividers)

    FEATURE (leave completely alone — full color, full depth):
      • Dark clusters that don't touch the background AND are ≥2px thick in
        both bbox dimensions — eyes, pupils, nose, spots, freckles

    Examples that work correctly:
      Outline ring    → touches bg                   → contour ✓
      Belt stripe     → connected to ring (bg) OR 1px thin → contour ✓
      2×2 eye cluster → no bg touch, min_dim=2       → feature ✓
      3×3 nose        → no bg touch, min_dim=3       → feature ✓
      Single dark dot → min_dim=1                    → contour (lone dots = line art) ✓
    """
    try:
        from scipy.ndimage import label, binary_dilation
    except ImportError:
        # Fallback: only replace silhouette pixels (bg-adjacent)
        is_dark    = detect_dark_pixels(colors, occupied)
        silhouette = detect_outlines(occupied)
        return is_dark & silhouette

    brightness = colors[:, :, :3].max(axis=2).astype(np.int32)
    is_dark    = occupied & (brightness < DARK_THRESHOLD)

    if not is_dark.any():
        return is_dark

    # Label connected dark regions with 8-connectivity
    struct8         = np.ones((3, 3), dtype=bool)
    labeled, n_comp = label(is_dark, structure=struct8)

    # Pre-compute which dark pixels are adjacent to background (8-connectivity)
    bg_dilated = binary_dilation(~occupied, structure=struct8)

    contour_mask = np.zeros_like(is_dark)
    for cid in range(1, n_comp + 1):
        comp = labeled == cid

        # Rule 1: touches background → definitely a contour outline
        if (comp & bg_dilated).any():
            contour_mask |= comp
            continue

        # Rule 2: bounding box thinner than 2px → interior line art (stripe/border)
        ys, xs  = np.where(comp)
        min_dim = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        if min_dim < 2:
            contour_mask |= comp
            # else: compact blob (eye, nose, spot) → leave as feature

    return contour_mask


def _blend_pixels(colors: np.ndarray, occupied: np.ndarray,
                  mask: np.ndarray, blend: float = 0.90) -> np.ndarray:
    """
    For each pixel in mask, replace its color with a blend toward the nearest
    bright (non-dark) occupied pixel.

    blend=0.90 → 90% nearest-bright color + 10% original dark.
    Gives a natural slightly-darkened surface tone at the former outline position
    rather than a hard edge or an abrupt jump to pure skin.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        brightness = colors[:, :, :3].max(axis=2).astype(np.int32)
        is_bright  = occupied & (brightness >= DARK_THRESHOLD)

        if not is_bright.any() or not mask.any():
            return colors

        _, nearest = distance_transform_edt(~is_bright, return_indices=True)
        result     = colors.copy()
        ys, xs     = np.where(mask & occupied)
        for y, x in zip(ys, xs):
            ny, nx = int(nearest[0][y, x]), int(nearest[1][y, x])
            near_c = colors[ny, nx].astype(float)
            orig_c = colors[y,  x].astype(float)
            result[y, x] = np.clip(
                near_c * blend + orig_c * (1.0 - blend),
                0, 255
            ).astype(np.uint8)
        return result
    except ImportError:
        return colors


def deoutline_sprite(sprite: Image.Image) -> Image.Image:
    """
    Replace CONTOUR dark pixels with a blend toward their nearest bright neighbor,
    while leaving ISOLATED dark clusters (eyes, nose, freckles, spots) completely
    untouched at their original color and depth.

    Contour = dark pixel touching the background or a bright region border
              (silhouette outline + region-boundary line art)
    Feature = dark pixel fully enclosed by other dark pixels
              (eyes, pupils, nose tip, spots — intentional surface detail)

    Like the LEGO corgi: the black nose bricks are full-depth and visible head-on;
    the body outline pixels are blended so the side view shows fur, not a black shell.

    The original sprite is untouched and used for the PDF front/back faces.
    """
    rgba     = np.array(sprite.convert("RGBA"), dtype=np.uint8)
    is_bg    = _bg_mask(rgba)
    occupied = ~is_bg
    colors   = rgba[:, :, :3].copy()

    # Identify which dark pixels are contour lines vs intentional features
    contour_mask = _detect_contour_darks(colors, occupied)

    if not contour_mask.any():
        return sprite

    # Blend contour pixels toward nearest bright neighbor
    colors = _blend_pixels(colors, occupied, contour_mask, blend=0.90)

    result = rgba.copy()
    result[:, :, :3] = colors
    return Image.fromarray(result, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Depth map  (interior only → propagate to dark/outline → smooth)
# ─────────────────────────────────────────────────────────────────────────────

def compute_depth_map(
    occupied:   np.ndarray,
    colors:     np.ndarray,
    region_map: np.ndarray,
    is_outline: np.ndarray,
    is_dark:    np.ndarray,
) -> np.ndarray:
    """
    Two-phase depth assignment
    ──────────────────────────
    Phase A  Interior (bright) pixels get depth from region + elliptical profile.
             Profile minimum = MIN_PROFILE (55 %) so sides are a smooth cylinder.

    Phase B  Dark / outline pixels inherit the depth of their nearest interior
             neighbour via distance transform.  They carry zero structural
             weight of their own — pure surface decoration.

    After both phases a Gaussian blur (sigma 1.2, 2 passes) smooths all
    remaining transitions so the face reads as one cohesive rounded mass.
    """
    H, W      = occupied.shape
    depth_map = np.zeros((H, W), dtype=np.float32)
    is_art    = (is_outline | is_dark) & occupied   # line-art mask
    interior  = occupied & ~is_art                   # structural pixels

    # ── Phase A: structural pixels ────────────────────────────────────────────
    for y in range(H):
        xs = np.where(interior[y, :])[0]
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs[0]), int(xs[-1])
        x_center     = (x_min + x_max) / 2.0
        half_w       = max(1.0, (x_max - x_min) / 2.0)

        for x in xs:
            region = int(region_map[y, x])
            max_d  = float(REGION_MAX_DEPTH.get(region, 5))
            t      = max(-1.0, min(1.0, (x - x_center) / half_w))
            profile         = math.sqrt(max(0.0, 1.0 - t * t))
            depth_map[y, x] = max(max_d * MIN_PROFILE,
                                  min(profile * max_d, float(MAX_DEPTH)))

    # ── Phase B: propagate interior depth to line-art pixels ─────────────────
    try:
        from scipy.ndimage import distance_transform_edt
        if interior.any():
            _, nearest = distance_transform_edt(~interior, return_indices=True)
            art_ys, art_xs = np.where(is_art)
            for y, x in zip(art_ys, art_xs):
                ny, nx          = int(nearest[0][y, x]), int(nearest[1][y, x])
                depth_map[y, x] = depth_map[ny, nx]
    except ImportError:
        for y in range(H):
            row_int = np.where(interior[y, :])[0]
            if len(row_int) == 0:
                continue
            row_max = float(depth_map[y, row_int].max())
            for x in np.where(is_art[y, :])[0]:
                depth_map[y, x] = row_max

    return np.round(depth_map).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Symmetry pass
# ─────────────────────────────────────────────────────────────────────────────

def apply_symmetry(
    depth_map:  np.ndarray,
    occupied:   np.ndarray,
    region_map: np.ndarray,
) -> np.ndarray:
    H, W   = depth_map.shape
    result = depth_map.copy()
    for y in range(H):
        xs = np.where(occupied[y, :])[0]
        if len(xs) < 2:
            continue
        x_center = (int(xs[0]) + int(xs[-1])) / 2.0
        for x in xs:
            if int(region_map[y, x]) in (REGION_LIMB, REGION_TUFT):
                continue
            x_m = int(round(2 * x_center - x))
            if 0 <= x_m < W and occupied[y, x_m]:
                avg              = (int(depth_map[y, x]) + int(depth_map[y, x_m])) // 2
                result[y, x]     = avg
                result[y, x_m]   = avg
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — VoxelGrid
# ─────────────────────────────────────────────────────────────────────────────

class VoxelGrid:
    def __init__(self, H: int, W: int, D: int):
        self.H, self.W, self.D = H, W, D
        self.occupied   = np.zeros((H, W, D), dtype=bool)
        self.colors     = np.zeros((H, W, D, 3), dtype=np.uint8)
        # viewer_colors: dark pixels replaced with bright neighbors — used for
        # ALL 3D renders (side, top, bottom, front in viewer).
        self.viewer_colors = np.zeros((H, W, D, 3), dtype=np.uint8)
        self.region_map = np.zeros((H, W), dtype=np.int8)

    @classmethod
    def build(cls, sprite: Image.Image) -> "VoxelGrid":
        # Resize first, then deoutline so resampling doesn't reintroduce darks.
        sprite_small = _resize(sprite.convert("RGBA"))
        sprite_vox   = deoutline_sprite(sprite_small)

        # Build from deoutlined sprite — no black pixels in the voxel grid at all
        rgba, occupied, colors = preprocess(sprite_vox)
        H, W = occupied.shape

        region_map = classify_regions(occupied)

        # Detect outlines/dark on the DEOUTLINED colors for depth computation.
        # Since deoutline already replaced darks, is_dark will be near-empty,
        # which means almost all pixels are treated as interior structure.
        # This is intentional: the deoutlined sprite has no line-art to exclude.
        is_outline = detect_outlines(occupied)
        is_dark    = detect_dark_pixels(colors, occupied)

        depth_map = compute_depth_map(occupied, colors, region_map, is_outline, is_dark)
        depth_map = apply_symmetry(depth_map, occupied, region_map)

        D    = min(int(depth_map.max()) if depth_map.max() > 0 else 1, MAX_DEPTH)
        grid = cls(H, W, D)
        grid.region_map = region_map

        # Store deoutlined colors in both color slots.
        # viewer_colors = what the 3D viewer uses (all bright, no black).
        # colors = same here since we build from the deoutlined sprite.
        # The PDF front face uses the ORIGINAL sprite (passed in papercraft.py),
        # so black line art is preserved there independently.
        ys, xs = np.where(occupied)
        for y, x in zip(ys, xs):
            d       = min(int(depth_map[y, x]), D)
            z_start = (D - d) // 2
            grid.occupied[y, x, z_start : z_start + d]        = True
            grid.colors[y, x, z_start : z_start + d]          = colors[y, x]
            grid.viewer_colors[y, x, z_start : z_start + d]   = colors[y, x]

        return grid

    def stats(self) -> dict:
        return {"grid_shape": (self.H, self.W, self.D),
                "voxel_count": int(self.occupied.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Surface rendering
# ─────────────────────────────────────────────────────────────────────────────

def _rep_color(colors_2d: np.ndarray, mask_1d: np.ndarray,
               min_bright: int = 65) -> np.ndarray:
    cands  = colors_2d[mask_1d].astype(np.float32)
    if len(cands) == 0:
        return np.array([180., 140., 80.])
    bright = cands[cands.max(axis=1) >= min_bright]
    if len(bright) == 0:
        bright = cands[cands.max(axis=1) >= 30]
    return np.median(bright if len(bright) else cands, axis=0)


def _to_pil(arr: np.ndarray, face_size: int, bg: tuple) -> Image.Image:
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")
    return img.resize((face_size, face_size), Image.NEAREST)


def _cull_surface(occ: np.ndarray, axis: int, from_back: bool) -> np.ndarray:
    shifted = np.roll(occ, 1 if from_back else -1, axis=axis)
    idx = [slice(None)] * 3
    idx[axis] = 0 if from_back else -1
    shifted[tuple(idx)] = False
    return occ & ~shifted


def _render_front_back(occ, col, shade, reverse_z, flip_x, face_size, bg):
    """Ray-cast along Z axis. Each pixel = color of first voxel hit."""
    H, W, D = occ.shape
    if reverse_z:
        occ = occ[:, :, ::-1]
        col = col[:, :, ::-1, :]

    any_hit = occ.any(axis=2)
    first_z = np.argmax(occ, axis=2)

    result  = np.full((H, W, 3), bg, dtype=np.float32)
    ys, xs  = np.where(any_hit)
    result[ys, xs] = col[ys, xs, first_z[ys, xs]].astype(np.float32)

    if flip_x:
        result = result[:, ::-1, :]

    return _to_pil((result * shade).clip(0, 255).astype(np.uint8), face_size, bg)


def _render_side(grid, is_left, shade, face_size, bg):
    """
    True orthographic ray-cast in the X direction.
    Uses viewer_colors (deoutlined, no black) for clean side renders.
    """
    occ = grid.occupied        # (H, W, D)
    col = grid.viewer_colors   # (H, W, D, 3) — always bright, no black voxels
    H, W, D = occ.shape

    x_order     = np.arange(W - 1, -1, -1) if not is_left else np.arange(W)
    occ_ordered = occ[:, x_order, :]
    col_ordered = col[:, x_order, :, :]

    any_hit = occ_ordered.any(axis=1)
    first_x = np.argmax(occ_ordered, axis=1)

    result = np.full((H, D, 3), bg, dtype=np.float32)
    ys, zs = np.where(any_hit)
    result[ys, zs] = col_ordered[ys, first_x[ys, zs], zs].astype(np.float32)

    t       = np.linspace(0, 1, D)
    result *= shade * (1.0 - 0.22 * t)[None, :, None]

    if not is_left:
        result = result[:, ::-1, :]

    return _to_pil(result.clip(0, 255).astype(np.uint8), face_size, bg)


def _render_top_bottom(grid, is_top, shade, face_size, bg):
    """
    True orthographic ray-cast in the Y direction.
    Uses viewer_colors (deoutlined, no black) for clean top/bottom renders.
    """
    occ = grid.occupied        # (H, W, D)
    col = grid.viewer_colors   # (H, W, D, 3) — always bright, no black voxels
    H, W, D = occ.shape

    y_order     = np.arange(H) if is_top else np.arange(H - 1, -1, -1)
    occ_ordered = occ[y_order, :, :]
    col_ordered = col[y_order, :, :, :]

    any_hit = occ_ordered.any(axis=0)
    first_y = np.argmax(occ_ordered, axis=0)

    result = np.full((D, W, 3), bg, dtype=np.float32)
    xs, zs = np.where(any_hit)
    result[zs, xs] = col_ordered[first_y[xs, zs], xs, zs].astype(np.float32)

    t       = np.linspace(0, 1, D)
    result *= shade * (1.0 - 0.18 * t)[:, None, None]

    if not is_top:
        result = result[::-1, :, :]

    return _to_pil(result.clip(0, 255).astype(np.uint8), face_size, bg)


def render_all_faces(grid, face_size=300, bg=(245, 245, 250)):
    occ = grid.occupied
    col = grid.viewer_colors   # use bright-only colors for all face renders
    return {
        "front":  _render_front_back(occ, col, 1.00, False, False, face_size, bg),
        "back":   _render_front_back(occ, col, 0.60, True,  True,  face_size, bg),
        "left":   _render_side(grid,  is_left=True,  shade=0.82, face_size=face_size, bg=bg),
        "right":  _render_side(grid,  is_left=False, shade=0.68, face_size=face_size, bg=bg),
        "top":    _render_top_bottom(grid, is_top=True,  shade=0.90, face_size=face_size, bg=bg),
        "bottom": _render_top_bottom(grid, is_top=False, shade=0.45, face_size=face_size, bg=bg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON bridge — Python → Three.js
# ─────────────────────────────────────────────────────────────────────────────

def get_depth_map(grid: VoxelGrid) -> np.ndarray:
    return grid.occupied.sum(axis=2).astype(np.int32)


def smooth_grid(grid: VoxelGrid, strength: float = 0.55, passes: int = 3) -> VoxelGrid:
    """
    Post-build Gaussian smoothing pass on the depth map.
    Preserves both colors and viewer_colors arrays through the rebuild.
    """
    H, W  = grid.H, grid.W
    depth = get_depth_map(grid).astype(np.float32)

    try:
        from scipy.ndimage import gaussian_filter
        for _ in range(passes):
            blurred = gaussian_filter(depth, sigma=0.6)
            depth   = np.where(depth > 0,
                               np.clip(blurred, 1.0, float(MAX_DEPTH)),
                               0.0)
        depth = np.round(depth).astype(np.int32)
    except ImportError:
        kern = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)
        for _ in range(passes):
            nxt = depth.copy()
            for y in range(H):
                for x in range(W):
                    if depth[y, x] == 0:
                        continue
                    ws = wt = 0.0
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            ny, nx = y+ky, x+kx
                            if 0 <= ny < H and 0 <= nx < W and depth[ny, nx] > 0:
                                w   = float(kern[ky+1, kx+1])
                                rw  = 1.0 if grid.region_map[ny,nx]==grid.region_map[y,x] else 0.35
                                ws += depth[ny, nx] * w * rw
                                wt += w * rw
                    if wt > 0:
                        sv = ws / wt
                        nxt[y,x] = max(1, min(MAX_DEPTH,
                                       round(depth[y,x]*(1-strength)+sv*strength)))
            depth = nxt

    new_D    = min(int(depth.max()), MAX_DEPTH) if depth.max() > 0 else 1
    new_grid = VoxelGrid(H, W, new_D)
    new_grid.region_map = grid.region_map.copy()
    ys, xs = np.where(depth > 0)
    for y, x in zip(ys, xs):
        d          = min(int(depth[y, x]), new_D)
        old_d      = int(grid.occupied[y, x, :].sum())
        old_zstart = (grid.D - old_d) // 2 if old_d > 0 else 0
        # Preserve both color arrays through smooth rebuild
        src_col = (grid.colors[y, x, old_zstart]
                   if old_d > 0 and old_zstart < grid.D
                   else np.array([180,140,80], dtype=np.uint8))
        src_vcol = (grid.viewer_colors[y, x, old_zstart]
                    if old_d > 0 and old_zstart < grid.D
                    else np.array([180,140,80], dtype=np.uint8))
        z_start = (new_D - d) // 2
        new_grid.occupied[y, x, z_start:z_start+d]      = True
        new_grid.colors[y, x, z_start:z_start+d]        = src_col
        new_grid.viewer_colors[y, x, z_start:z_start+d] = src_vcol
    return new_grid


def export_grid_json(grid: VoxelGrid) -> dict:
    """
    Serialise to JSON for Three.js.
    Uses viewer_colors (deoutlined, all bright) so the Three.js viewer
    never renders black voxels — even when rotating to look at the sides.
    """
    H, W   = grid.H, grid.W
    depth  = get_depth_map(grid)

    # Build front-color array from viewer_colors (bright, no black)
    color_map, region_map = [], []
    for y in range(H):
        crow, rrow = [], []
        for x in range(W):
            if depth[y, x] > 0:
                occ_zs = np.where(grid.occupied[y, x, :])[0]
                z0     = int(occ_zs[0]) if len(occ_zs) > 0 else 0
                c      = grid.viewer_colors[y, x, z0]
                crow.append([int(c[0]), int(c[1]), int(c[2])])
                rrow.append(int(grid.region_map[y, x]))
            else:
                crow.append(None)
                rrow.append(None)
        color_map.append(crow)
        region_map.append(rrow)

    return {
        "H":         H,
        "W":         W,
        "MAX_D":     int(grid.D),
        "depthMap":  depth.tolist(),
        "colorMap":  color_map,
        "regionMap": region_map,
    }
