"""
voxelizer.py — Full voxelization pipeline for pixel-art sprites.

Design philosophy (chibi / Lego-style sculpt)
─────────────────────────────────────────────
• Outline / dark pixels are VISUAL LINE ART only — they inherit their depth
  from the nearest bright interior pixel via distance transform.  They never
  contribute structural depth data of their own.
• Depth contrast is deliberately low.  The face is a smooth rounded mass,
  not a cratered landscape.  Facial features sit ON the surface.
• Region depths differ meaningfully (HEAD=14, TORSO=11, LIMB=6) so each
  body segment is visually distinct on the side view — head pops out most,
  torso is a clearly shallower block, limbs are thin.
• Elliptical profile minimum is 55 % so the silhouette reads as a rounded
  cylinder with real edge depth, not a pointed cone.  This prevents edge
  pixels (including eyes near the silhouette) from popping out relative
  to their neighbours.
• Region boundaries use NO seam scaling — the depth step between regions
  is enough to read the segments without cutting grooves that slice through
  features at the head/torso boundary.
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

REGION_MAX_DEPTH: dict[int, int] = {
    REGION_HEAD:  14,
    REGION_TORSO: 11,
    REGION_LIMB:   6,
    REGION_TUFT:   3,
}
MAX_DEPTH      = 14
DARK_THRESHOLD = 55
MIN_PROFILE    = 0.55

# Fallback color used when a smoothed pixel has no source voxel color.
_FALLBACK_COLOR = np.array([180, 140, 80], dtype=np.uint8)

REGION_CODES = {
    "head":  REGION_HEAD,
    "torso": REGION_TORSO,
    "limb":  REGION_LIMB,
    "tuft":  REGION_TUFT,
}
REGION_NAMES = {v: k for k, v in REGION_CODES.items()}


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
    """
    Assign each pixel to HEAD, TORSO, LIMB, or TUFT based on vertical position
    within the character's bounding box.

    Position-based heuristic (more reliable than seam detection for chibi sprites
    where head and torso are often connected with no gap):
      • Top narrow rows (width < 40% of max) → TUFT (hat brim, antenna, hair tuft)
      • Remaining span split: top 42% → HEAD, next 36% → TORSO, bottom 22% → LIMB
    """
    H, W       = occupied.shape
    region_map = np.full((H, W), REGION_LIMB, dtype=np.int8)
    row_counts = occupied.sum(axis=1).astype(float)
    max_w      = max(float(row_counts.max()), 1.0)

    occ_rows = np.where(row_counts > 0)[0]
    if len(occ_rows) == 0:
        return region_map
    y_top, y_bot = int(occ_rows[0]), int(occ_rows[-1])

    y_head_start = y_top
    for y in range(y_top, H):
        if row_counts[y] < max_w * 0.40:
            region_map[y, :] = REGION_TUFT
            y_head_start = y + 1
        else:
            break

    body_span = max(1, y_bot - y_head_start + 1)
    head_end  = y_head_start + int(body_span * 0.42)
    torso_end = y_head_start + int(body_span * 0.78)

    for y in range(y_head_start, y_bot + 1):
        if row_counts[y] == 0:
            continue
        if y < head_end:
            region_map[y, :] = REGION_HEAD
        elif y < torso_end:
            region_map[y, :] = REGION_TORSO
        else:
            region_map[y, :] = REGION_LIMB

    return region_map


def scale_region_map(region_map_src: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Scale a region map to a different resolution using nearest-neighbour mapping.
    Vectorized with numpy indexing instead of Python loops.
    """
    src_h, src_w = region_map_src.shape
    # Build index arrays for the nearest source pixel at each target coordinate.
    ys = np.clip((np.arange(target_h) / target_h * src_h).astype(int), 0, src_h - 1)
    xs = np.clip((np.arange(target_w) / target_w * src_w).astype(int), 0, src_w - 1)
    # Fancy indexing: out[y, x] = src[ys[y], xs[x]]
    return region_map_src[np.ix_(ys, xs)].copy()


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

    CONTOUR → replace color with nearest bright neighbor:
      • Touches the background (silhouette outline)
      • Bounding box is only 1 px thick in either dim (interior border stripes)

    FEATURE → leave completely alone:
      • Dark clusters that don't touch background AND are ≥2px thick in both dims
        (eyes, pupils, nose, spots, freckles)
    """
    try:
        from scipy.ndimage import label, binary_dilation
    except ImportError:
        is_dark    = detect_dark_pixels(colors, occupied)
        silhouette = detect_outlines(occupied)
        return is_dark & silhouette

    brightness = colors[:, :, :3].max(axis=2).astype(np.int32)
    is_dark    = occupied & (brightness < DARK_THRESHOLD)

    if not is_dark.any():
        return is_dark

    struct8         = np.ones((3, 3), dtype=bool)
    labeled, n_comp = label(is_dark, structure=struct8)
    bg_dilated      = binary_dilation(~occupied, structure=struct8)

    contour_mask = np.zeros_like(is_dark)
    for cid in range(1, n_comp + 1):
        comp = labeled == cid
        if (comp & bg_dilated).any():
            contour_mask |= comp
            continue
        ys, xs  = np.where(comp)
        min_dim = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        if min_dim < 2:
            contour_mask |= comp

    return contour_mask


def _blend_pixels(colors: np.ndarray, occupied: np.ndarray,
                  mask: np.ndarray, blend: float = 0.90) -> np.ndarray:
    """
    For each pixel in mask, replace its color with a blend toward the nearest
    bright (non-dark) occupied pixel.  Vectorized with fancy indexing.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        brightness = colors[:, :, :3].max(axis=2).astype(np.int32)
        is_bright  = occupied & (brightness >= DARK_THRESHOLD)

        if not is_bright.any() or not mask.any():
            return colors

        _, nearest = distance_transform_edt(~is_bright, return_indices=True)
        result     = colors.copy()

        # Vectorized: get all (y,x) positions to update at once.
        ys, xs = np.where(mask & occupied)
        if len(ys) == 0:
            return result
        nys = nearest[0][ys, xs]
        nxs = nearest[1][ys, xs]
        near_c = colors[nys, nxs].astype(np.float32)
        orig_c = colors[ys,  xs ].astype(np.float32)
        blended = np.clip(near_c * blend + orig_c * (1.0 - blend), 0, 255).astype(np.uint8)
        result[ys, xs] = blended
        return result
    except ImportError:
        return colors


def deoutline_sprite(sprite: Image.Image) -> Image.Image:
    """
    Replace CONTOUR dark pixels with a blend toward their nearest bright neighbor,
    while leaving ISOLATED dark clusters (eyes, nose, freckles) untouched.
    """
    rgba     = np.array(sprite.convert("RGBA"), dtype=np.uint8)
    is_bg    = _bg_mask(rgba)
    occupied = ~is_bg
    colors   = rgba[:, :, :3].copy()

    contour_mask = _detect_contour_darks(colors, occupied)

    if not contour_mask.any():
        return sprite

    colors = _blend_pixels(colors, occupied, contour_mask, blend=0.90)

    result = rgba.copy()
    result[:, :, :3] = colors
    return Image.fromarray(result, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Depth map
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
    Phase B  Dark / outline pixels inherit depth from nearest interior neighbour.
    """
    H, W      = occupied.shape
    depth_map = np.zeros((H, W), dtype=np.float32)
    is_art    = (is_outline | is_dark) & occupied
    interior  = occupied & ~is_art

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

    try:
        from scipy.ndimage import distance_transform_edt
        if interior.any():
            _, nearest = distance_transform_edt(~interior, return_indices=True)
            art_ys, art_xs = np.where(is_art)
            if len(art_ys) > 0:
                # Vectorized propagation
                nys = nearest[0][art_ys, art_xs]
                nxs = nearest[1][art_ys, art_xs]
                depth_map[art_ys, art_xs] = depth_map[nys, nxs]
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
        self.occupied      = np.zeros((H, W, D), dtype=bool)
        self.colors        = np.zeros((H, W, D, 3), dtype=np.uint8)
        self.viewer_colors = np.zeros((H, W, D, 3), dtype=np.uint8)
        self.region_map    = np.zeros((H, W), dtype=np.int8)

    @classmethod
    def build(
        cls,
        sprite: Image.Image,
        external_region_map: np.ndarray | None = None,
    ) -> "VoxelGrid":
        sprite_small = _resize(sprite.convert("RGBA"))
        sprite_vox   = deoutline_sprite(sprite_small)

        rgba, occupied, colors = preprocess(sprite_vox)
        H, W = occupied.shape

        if external_region_map is not None:
            region_map = scale_region_map(external_region_map, H, W)
            region_map = np.where(occupied, region_map, REGION_LIMB).astype(np.int8)
        else:
            region_map = classify_regions(occupied)

        is_outline = detect_outlines(occupied)
        is_dark    = detect_dark_pixels(colors, occupied)

        depth_map = compute_depth_map(occupied, colors, region_map, is_outline, is_dark)
        depth_map = apply_symmetry(depth_map, occupied, region_map)

        D    = min(int(depth_map.max()) if depth_map.max() > 0 else 1, MAX_DEPTH)
        grid = cls(H, W, D)
        grid.region_map = region_map

        # Vectorized voxel fill: compute z_start for every occupied pixel at once.
        ys, xs = np.where(occupied)
        depths = np.minimum(depth_map[ys, xs], D)
        z_starts = (D - depths) // 2
        for i, (y, x) in enumerate(zip(ys, xs)):
            d  = int(depths[i])
            z0 = int(z_starts[i])
            grid.occupied[y, x, z0:z0+d]        = True
            grid.colors[y, x, z0:z0+d]          = colors[y, x]
            grid.viewer_colors[y, x, z0:z0+d]   = colors[y, x]

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


def _render_front_back(occ, col, shade, reverse_z, flip_x, face_size, bg):
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
    occ = grid.occupied
    col = grid.viewer_colors
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
    occ = grid.occupied
    col = grid.viewer_colors
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
    col = grid.viewer_colors
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


def smooth_grid(grid: VoxelGrid, passes: int = 2) -> VoxelGrid:
    """
    Region-aware smoothing: blurs depth within each region for organic roundness,
    but does NOT blur across region boundaries so head/torso/limb depth steps
    remain sharp and readable.

    Uses scipy.ndimage convolution per region for ~100x speedup over the
    previous pure-Python nested loop approach.
    """
    H, W  = grid.H, grid.W
    depth = get_depth_map(grid).astype(np.float32)
    rmap  = grid.region_map

    try:
        from scipy.ndimage import uniform_filter

        # Process each region independently to avoid blurring across boundaries.
        for _ in range(passes):
            nxt = depth.copy()
            for region_id in (REGION_HEAD, REGION_TORSO, REGION_LIMB, REGION_TUFT):
                mask = (rmap == region_id) & (depth > 0)
                if not mask.any():
                    continue
                # Smooth only within this region: zero out other regions, blur,
                # then blend back only at region pixels.
                region_depth = np.where(mask, depth, 0.0)
                # uniform_filter smooths including border zeros — we correct by
                # dividing by the fraction of the kernel that was non-zero.
                region_smooth = uniform_filter(region_depth, size=3, mode='constant', cval=0.0)
                weight_map    = uniform_filter(mask.astype(np.float32), size=3, mode='constant', cval=0.0)
                # Avoid division by zero; pixels with no in-region neighbours keep original.
                safe_w = np.where(weight_map > 0, weight_map, 1.0)
                corrected = region_smooth / safe_w
                # Blend: 40% original, 60% smoothed — same ratio as before.
                blended = np.clip(depth * 0.4 + corrected * 0.6, 1.0, float(MAX_DEPTH))
                nxt = np.where(mask, blended, nxt)
            depth = nxt

    except ImportError:
        # Fallback: original Python loop implementation
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
                            ny, nx2 = y + ky, x + kx
                            if 0 <= ny < H and 0 <= nx2 < W and depth[ny, nx2] > 0:
                                if rmap[ny, nx2] == rmap[y, x]:
                                    w   = float(kern[ky+1, kx+1])
                                    ws += depth[ny, nx2] * w
                                    wt += w
                    if wt > 0:
                        nxt[y, x] = max(1.0, min(float(MAX_DEPTH),
                                                 depth[y, x] * 0.4 + (ws / wt) * 0.6))
            depth = nxt

    depth = np.round(depth).astype(np.int32)
    new_D    = min(int(depth.max()), MAX_DEPTH) if depth.max() > 0 else 1
    new_grid = VoxelGrid(H, W, new_D)
    new_grid.region_map = grid.region_map.copy()

    ys, xs = np.where(depth > 0)
    # Batch-fetch source colors: use the first occupied z-slice per pixel.
    old_occ_count = grid.occupied.sum(axis=2)   # H×W
    old_zstart    = np.argmax(grid.occupied, axis=2)  # H×W (0 when no voxel)

    for i, (y, x) in enumerate(zip(ys, xs)):
        d          = min(int(depth[y, x]), new_D)
        old_d      = int(old_occ_count[y, x])
        old_z0     = int(old_zstart[y, x]) if old_d > 0 else 0
        src_col    = grid.colors[y, x, old_z0]       if old_d > 0 else _FALLBACK_COLOR
        src_vcol   = grid.viewer_colors[y, x, old_z0] if old_d > 0 else _FALLBACK_COLOR
        z_start = (new_D - d) // 2
        new_grid.occupied[y, x, z_start:z_start+d]      = True
        new_grid.colors[y, x, z_start:z_start+d]        = src_col
        new_grid.viewer_colors[y, x, z_start:z_start+d] = src_vcol

    return new_grid


def export_grid_json(grid: VoxelGrid) -> dict:
    """
    Serialise to JSON for Three.js.
    Uses viewer_colors (deoutlined, all bright) so the viewer never renders
    black voxels.
    """
    H, W   = grid.H, grid.W
    depth  = get_depth_map(grid)

    # Vectorized: find first occupied z-layer per pixel.
    any_occ  = depth > 0
    first_z  = np.argmax(grid.occupied, axis=2)  # H×W

    color_map, region_map_out = [], []
    for y in range(H):
        crow, rrow = [], []
        for x in range(W):
            if any_occ[y, x]:
                c = grid.viewer_colors[y, x, first_z[y, x]]
                crow.append([int(c[0]), int(c[1]), int(c[2])])
                rrow.append(int(grid.region_map[y, x]))
            else:
                crow.append(None)
                rrow.append(None)
        color_map.append(crow)
        region_map_out.append(rrow)

    return {
        "H":         H,
        "W":         W,
        "MAX_D":     int(grid.D),
        "depthMap":  depth.tolist(),
        "colorMap":  color_map,
        "regionMap": region_map_out,
    }
