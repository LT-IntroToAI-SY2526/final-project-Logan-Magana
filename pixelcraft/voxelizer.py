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
• Color blending (dark → nearest interior) is applied ONLY when exporting
  the Three.js JSON, so the papercraft PDF front face retains correct
  black line art while the 3-D viewer never shows black shells.
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

# Chibi proportions: head ≈ torso mass, limbs thin
REGION_MAX_DEPTH: dict[int, int] = {
    REGION_HEAD:  16,
    REGION_TORSO: 14,
    REGION_LIMB:  8,
}
MAX_DEPTH      = 16   # deeper extrusion — side profile has real sculptural mass
DARK_THRESHOLD = 55   # max(r,g,b) below this → treat as line art, not structure
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

    return region_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Outline + dark pixel detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outlines(occupied: np.ndarray) -> np.ndarray:
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
            # sqrt gives rounded cylinder; floor at MIN_PROFILE for stable sides
            profile         = math.sqrt(max(0.0, 1.0 - t * t))
            depth_map[y, x] = max(max_d * MIN_PROFILE,
                                  min(profile * max_d, float(MAX_DEPTH)))

    # ── Phase B: propagate interior depth to line-art pixels ─────────────────
    try:
        from scipy.ndimage import distance_transform_edt
        if interior.any():
            # For every art pixel, find the nearest interior pixel and copy depth
            _, nearest = distance_transform_edt(~interior, return_indices=True)
            art_ys, art_xs = np.where(is_art)
            for y, x in zip(art_ys, art_xs):
                ny, nx          = int(nearest[0][y, x]), int(nearest[1][y, x])
                depth_map[y, x] = depth_map[ny, nx]
    except ImportError:
        # Fallback: use row-maximum of interior depth
        for y in range(H):
            row_int = np.where(interior[y, :])[0]
            if len(row_int) == 0:
                continue
            row_max = float(depth_map[y, row_int].max())
            for x in np.where(is_art[y, :])[0]:
                depth_map[y, x] = row_max

    # ── Gaussian smooth: erase residual cliff edges ──────────────────────────
    try:
        from scipy.ndimage import gaussian_filter
        for _ in range(2):
            blurred         = gaussian_filter(depth_map, sigma=0.8)
            # Only update occupied pixels; preserve zeros in background
            depth_map       = np.where(occupied, blurred, 0.0)
            # Clamp to valid range
            depth_map       = np.where(occupied,
                                       np.clip(depth_map, 1.0, float(MAX_DEPTH)),
                                       0.0)
    except ImportError:
        pass

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
            if int(region_map[y, x]) == REGION_LIMB:
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
        self.region_map = np.zeros((H, W), dtype=np.int8)

    @classmethod
    def build(cls, sprite: Image.Image) -> "VoxelGrid":
        rgba, occupied, colors = preprocess(sprite)
        H, W = occupied.shape

        region_map = classify_regions(occupied)
        is_outline = detect_outlines(occupied)
        is_dark    = detect_dark_pixels(colors, occupied)

        # Colors stored as-is (original line art preserved for PDF front face).
        # Color blending for the 3-D viewer is applied later in export_grid_json.
        depth_map = compute_depth_map(occupied, colors, region_map, is_outline, is_dark)
        depth_map = apply_symmetry(depth_map, occupied, region_map)

        D    = min(int(depth_map.max()) if depth_map.max() > 0 else 1, MAX_DEPTH)
        grid = cls(H, W, D)
        grid.region_map = region_map

        # All pixels use bilateral centering — since dark/outline pixels now
        # have depths matching their neighbours, there are no cliff edges and
        # therefore no protrusions or hollows at the surface.
        ys, xs = np.where(occupied)
        for y, x in zip(ys, xs):
            d       = min(int(depth_map[y, x]), D)
            z_start = (D - d) // 2
            grid.occupied[y, x, z_start : z_start + d] = True
            grid.colors[y, x, z_start : z_start + d]   = colors[y, x]

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
    H, W, D = occ.shape
    if reverse_z:
        occ = occ[:, :, ::-1]
        col = col[:, :, ::-1, :]
    surface = _cull_surface(occ, axis=2, from_back=False)
    any_hit = surface.any(axis=2)
    first_z = np.argmax(surface, axis=2)
    result  = np.full((H, W, 3), bg, dtype=np.float32)
    ys, xs  = np.where(any_hit)
    result[ys, xs] = col[ys, xs, first_z[ys, xs]].astype(np.float32)
    if flip_x:
        result = result[:, ::-1, :]
    return _to_pil((result * shade).astype(np.uint8), face_size, bg)


def _render_side(grid, is_left, shade, face_size, bg):
    occ = grid.occupied
    col = grid.colors
    H, W, D = occ.shape
    surface_x = (_cull_surface(occ, axis=1, from_back=False) if is_left
                 else _cull_surface(occ, axis=1, from_back=True))
    col_front = col[:, :, 0, :]
    occ_front = occ[:, :, 0]
    result = np.full((H, D, 3), bg, dtype=np.float32)
    for y in range(H):
        row_mask = occ_front[y, :]
        if not row_mask.any():
            continue
        rep       = _rep_color(col_front[y], row_mask, min_bright=65)
        z_surface = surface_x[y, :, :].any(axis=0)
        if not z_surface.any():
            z_surface = occ[y, :, :].any(axis=0)
        if not z_surface.any():
            continue
        for z in range(D):
            if z_surface[z]:
                t_depth         = z / max(1, D - 1)
                result[y, z, :] = rep * shade * (1.0 - 0.25 * t_depth)
    if not is_left:
        result = result[:, ::-1, :]
    return _to_pil(result.astype(np.uint8), face_size, bg)


def _render_top_bottom(grid, is_top, shade, face_size, bg):
    occ = grid.occupied
    col = grid.colors
    H, W, D = occ.shape
    surface_y = (_cull_surface(occ, axis=0, from_back=False) if is_top
                 else _cull_surface(occ, axis=0, from_back=True))
    col_front = col[:, :, 0, :]
    occ_front = occ[:, :, 0]
    result = np.full((D, W, 3), bg, dtype=np.float32)
    for x in range(W):
        col_mask = occ_front[:, x]
        if not col_mask.any():
            continue
        top_col = None
        y_range = range(H) if is_top else range(H - 1, -1, -1)
        for y in y_range:
            if col_mask[y] and col_front[y, x, :].max() >= 65:
                top_col = col_front[y, x, :].astype(np.float32)
                break
        if top_col is None:
            top_col = _rep_color(col_front[:, x], col_mask, min_bright=65)
        z_surface = surface_y[:, x, :].any(axis=0)
        if not z_surface.any():
            z_surface = occ[:, x, :].any(axis=0)
        if not z_surface.any():
            continue
        for z in range(D):
            if z_surface[z]:
                t_depth      = z / max(1, D - 1)
                result[z, x, :] = top_col * shade * (1.0 - 0.20 * t_depth)
    if not is_top:
        result = result[::-1, :, :]
    return _to_pil(result.astype(np.uint8), face_size, bg)


def render_all_faces(grid, face_size=300, bg=(245, 245, 250)):
    occ = grid.occupied
    col = grid.colors
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
    The gaussian_filter in compute_depth_map already does the heavy lifting;
    this adds a final light pass to blend any seam-region boundaries.
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
        # Manual 3×3 fallback
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
        src = (grid.colors[y, x, old_zstart]
               if old_d > 0 and old_zstart < grid.D
               else np.array([180,140,80], dtype=np.uint8))
        z_start = (new_D - d) // 2
        new_grid.occupied[y, x, z_start:z_start+d] = True
        new_grid.colors[y, x, z_start:z_start+d]   = src
    return new_grid


def _blend_for_viewer(colors: np.ndarray, occupied: np.ndarray,
                      is_dark: np.ndarray, blend: float = 0.78) -> np.ndarray:
    """
    Blend dark pixel colors toward nearest interior pixel — for viewer only.
    Original colors in the grid are untouched so the PDF front face keeps
    correct black line art.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        interior = occupied & ~is_dark
        if not interior.any():
            return colors
        _, nearest = distance_transform_edt(~interior, return_indices=True)
        result     = colors.copy()
        ys, xs     = np.where(is_dark & occupied)
        for y, x in zip(ys, xs):
            ny, nx       = int(nearest[0][y, x]), int(nearest[1][y, x])
            result[y, x] = np.clip(
                colors[ny, nx].astype(float) * blend +
                colors[y,  x].astype(float) * (1.0 - blend),
                0, 255
            ).astype(np.uint8)
        return result
    except ImportError:
        return colors


def export_grid_json(grid: VoxelGrid) -> dict:
    """
    Serialise to JSON for Three.js.  Color blending (dark → interior) is
    applied here only — the grid itself retains original sprite colors.
    """
    H, W   = grid.H, grid.W
    depth  = get_depth_map(grid)

    # Build front-color array from first occupied z-slice per pixel
    front_colors = np.zeros((H, W, 3), dtype=np.uint8)
    front_occ    = (depth > 0)
    for y in range(H):
        for x in range(W):
            if depth[y, x] > 0:
                occ_zs = np.where(grid.occupied[y, x, :])[0]
                z0     = int(occ_zs[0]) if len(occ_zs) > 0 else 0
                front_colors[y, x] = grid.colors[y, x, z0]

    # Selective color blend: ONLY border outline pixels (pixels touching the
    # background silhouette edge) get blended toward nearest interior color.
    # Interior dark pixels (eyes, hair, facial features) keep original color.
    # This removes the black column on side faces while preserving eye clarity.
    is_border = detect_outlines(front_occ)   # True only for silhouette-edge pixels
    blended   = _blend_for_viewer(front_colors, front_occ, is_border, blend=0.80)

    color_map, region_map = [], []
    for y in range(H):
        crow, rrow = [], []
        for x in range(W):
            if depth[y, x] > 0:
                c = blended[y, x]
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
