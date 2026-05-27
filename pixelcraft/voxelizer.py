"""
voxelizer.py — Full voxelization pipeline for pixel-art sprites.

Pipeline
────────
1.  Pre-process       : cap size, detect background, extract silhouette + colors.
2.  Region classify   : HEAD / TORSO / LIMB rows via seam + width analysis.
3.  Outline detect    : border pixels → thin (depth=1) voxel columns.
4.  Depth map         : region-based fixed max-depth + elliptical profile.
                        HEAD=8  TORSO=10  LIMB=6  outline=1
                        Depth is NOT row-width-relative — avoids the cone artifact.
5.  Symmetry pass     : mirror depth across the sprite centre-line (humanoids).
6.  VoxelGrid         : fill 3-D occupancy + color arrays (H × W × D).
7.  Surface rendering : hidden-face culling + filled-column side/top renders.
                        Only surface voxels (exposed face in view direction)
                        are included.  Filled approach eliminates internal
                        cross-section streaks.
"""

from __future__ import annotations
import math
import numpy as np
from PIL import Image

# ── tunables ──────────────────────────────────────────────────────────────────
MAX_DIM = 64          # sprite capped at this before voxelization

# Region codes
REGION_HEAD  = 0
REGION_TORSO = 1
REGION_LIMB  = 2

# Fixed max-depth per region (independent of row width — prevents cone artifact)
REGION_MAX_DEPTH: dict[int, int] = {
    REGION_HEAD:  8,
    REGION_TORSO: 10,
    REGION_LIMB:  6,
}
MAX_DEPTH = max(REGION_MAX_DEPTH.values())   # = 10


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Pre-process
# ─────────────────────────────────────────────────────────────────────────────

def _resize(sprite: Image.Image) -> Image.Image:
    w, h = sprite.size
    if max(w, h) > MAX_DIM:
        s = MAX_DIM / max(w, h)
        sprite = sprite.resize((max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)
    return sprite


def _bg_mask(rgba: np.ndarray, tol: int = 30) -> np.ndarray:
    """Return bool (H, W) True = background."""
    H, W  = rgba.shape[:2]
    alpha = rgba[:, :, 3]
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

    # Ensure topmost quarter of first wide segment is still HEAD
    first = segments[0]
    if first[2] >= widest * 0.75:
        he = first[0] + max(1, (first[1] - first[0]) // 4)
        region_map[first[0]:he, :] = REGION_HEAD

    return region_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Outline detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outlines(occupied: np.ndarray) -> np.ndarray:
    """True where pixel borders at least one background cell (4-connectivity)."""
    H, W       = occupied.shape
    is_outline = np.zeros((H, W), dtype=bool)
    padded     = np.pad(occupied, 1, constant_values=False)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        nb = padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        is_outline |= (occupied & ~nb)
    return is_outline


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Depth map  (region-fixed base + elliptical horizontal profile)
# ─────────────────────────────────────────────────────────────────────────────

def compute_depth_map(
    occupied:   np.ndarray,
    colors:     np.ndarray,
    region_map: np.ndarray,
    is_outline: np.ndarray,
) -> np.ndarray:
    """
    Depth rules
    -----------
    outline pixels  → 1   (thin border shell)
    interior pixels → REGION_MAX_DEPTH[region] × elliptical_profile(x_pos)

    The elliptical profile rounds the HORIZONTAL cross-section (makes the
    model look round when viewed from above / sides) without varying the
    maximum depth per row — that is fixed per region, preventing the cone
    artifact that comes from depth ∝ row_width.
    """
    H, W      = occupied.shape
    depth_map = np.zeros((H, W), dtype=np.int32)

    for y in range(H):
        xs = np.where(occupied[y, :])[0]
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs[0]), int(xs[-1])
        x_center     = (x_min + x_max) / 2.0
        half_w       = max(1.0, (x_max - x_min) / 2.0)

        for x in xs:
            if is_outline[y, x]:
                depth_map[y, x] = 1
                continue

            region = int(region_map[y, x])
            max_d  = REGION_MAX_DEPTH.get(region, 6)

            # Elliptical profile: 1.0 at centre, 0 at edge
            t       = max(-1.0, min(1.0, (x - x_center) / half_w))
            profile = math.sqrt(max(0.0, 1.0 - t * t))

            # At least 40 % of max_d even at the edges (avoids razor-thin sides)
            depth_map[y, x] = max(max(1, int(max_d * 0.40)),
                                  min(int(profile * max_d), MAX_DEPTH))

    return depth_map


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
                avg = (int(depth_map[y, x]) + int(depth_map[y, x_m])) // 2
                result[y, x]   = avg
                result[y, x_m] = avg
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — VoxelGrid
# ─────────────────────────────────────────────────────────────────────────────

class VoxelGrid:
    """
    3-D voxel grid.
    Axes: Y=row (top→bottom), X=col (left→right), Z=depth (0=front).
    Also stores region_map and per-row/column representative colors for rendering.
    """

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
        depth_map  = compute_depth_map(occupied, colors, region_map, is_outline)
        depth_map  = apply_symmetry(depth_map, occupied, region_map)

        D    = min(int(depth_map.max()) if depth_map.max() > 0 else 1, MAX_DEPTH)
        grid = cls(H, W, D)
        grid.region_map = region_map

        ys, xs = np.where(occupied)
        for y, x in zip(ys, xs):
            d       = min(int(depth_map[y, x]), D)
            # ── Bilateral symmetric extrusion ─────────────────────────────
            # Centre each column in Z so the model is symmetric front-to-back.
            # A head pixel with d=8 in a D=10 grid gets z_start=1, voxels at
            # z=1..8.  An outline pixel (d=1) gets z_start=4, centred at z=4.
            # This makes the 3-D silhouette spherical/cylindrical rather than
            # a flat slab.
            z_start = (D - d) // 2
            grid.occupied[y, x, z_start : z_start + d] = True
            grid.colors[y, x, z_start : z_start + d]   = colors[y, x]

        return grid

    def stats(self) -> dict:
        return {"grid_shape": (self.H, self.W, self.D),
                "voxel_count": int(self.occupied.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Surface-only rendering with hidden face culling
# ─────────────────────────────────────────────────────────────────────────────

def _rep_color(colors_2d: np.ndarray, mask_1d: np.ndarray,
               min_bright: int = 40) -> np.ndarray:
    """
    Return the median non-dark color of a 1-D set of pixels.
    `colors_2d[mask_1d]` gives the candidate pixels.
    Falls back to full median if nothing is bright enough.
    """
    cands = colors_2d[mask_1d].astype(np.float32)   # (N, 3)
    if len(cands) == 0:
        return np.array([180., 140., 80.])
    bright = cands[cands.max(axis=1) >= min_bright]
    return np.median(bright if len(bright) else cands, axis=0)


def _to_pil(arr: np.ndarray, face_size: int, bg: tuple) -> Image.Image:
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")
    return img.resize((face_size, face_size), Image.NEAREST)


def _cull_surface(occ: np.ndarray, axis: int, from_back: bool) -> np.ndarray:
    """
    Hidden-face culling: return bool mask of voxels whose face on `axis`
    (looking from front=0 or back) is EXPOSED to air.

    A voxel v has an exposed face in the +axis direction iff the neighbour
    in that direction is empty (or out-of-bounds).
    """
    # Shift the occupancy array by 1 along `axis`
    shifted = np.roll(occ, 1 if from_back else -1, axis=axis)
    # At the boundary the roll wraps — zero it out
    if from_back:
        idx = [slice(None)] * 3
        idx[axis] = 0
        shifted[tuple(idx)] = False
    else:
        idx = [slice(None)] * 3
        idx[axis] = -1
        shifted[tuple(idx)] = False
    # Exposed = occupied AND neighbour in that direction is NOT occupied
    return occ & ~shifted


def _render_front_back(
    occ: np.ndarray, col: np.ndarray,
    shade: float, reverse_z: bool, flip_x: bool,
    face_size: int, bg: tuple,
) -> Image.Image:
    """
    Front / back: ray-cast along Z, show actual sprite colors.
    Surface culled: only pixels whose Z-facing surface is exposed.
    """
    H, W, D = occ.shape
    if reverse_z:
        occ = occ[:, :, ::-1]
        col = col[:, :, ::-1, :]

    # Surface mask: voxels exposed on the near Z face
    surface = _cull_surface(occ, axis=2, from_back=False)   # (H,W,D)

    # For each (y,x) find the first surface voxel along Z
    any_hit  = surface.any(axis=2)
    first_z  = np.argmax(surface, axis=2)
    result   = np.full((H, W, 3), bg, dtype=np.float32)
    ys, xs   = np.where(any_hit)
    result[ys, xs] = col[ys, xs, first_z[ys, xs]].astype(np.float32)

    if flip_x:
        result = result[:, ::-1, :]

    return _to_pil((result * shade).astype(np.uint8), face_size, bg)


def _render_side(
    grid: VoxelGrid,
    is_left: bool,
    shade: float,
    face_size: int,
    bg: tuple,
) -> Image.Image:
    """
    Left / right face — filled-column renderer with hidden face culling.

    For each sprite row y:
      1. Find which Z slices have a surface voxel on the X-facing side.
         Surface X voxel: occupied[y,x,z] AND NOT occupied[y, x±1, z].
      2. Fill those (y, z) pixels with the row's representative bright color.
      3. Apply depth gradient (darker toward back) for the rounded feel.

    This eliminates internal cross-section streaks because we never look
    THROUGH the model — only surface voxels contribute color.
    """
    occ = grid.occupied   # (H, W, D)
    col = grid.colors     # (H, W, D, 3)
    H, W, D = occ.shape

    # Surface culling along X axis
    if is_left:
        surface_x = _cull_surface(occ, axis=1, from_back=False)  # exposed LEFT  face
    else:
        surface_x = _cull_surface(occ, axis=1, from_back=True)   # exposed RIGHT face

    # Front sprite colors (z=0) for representative color lookup
    col_front = col[:, :, 0, :]   # (H, W, 3)
    occ_front = occ[:, :, 0]      # (H, W)

    result = np.full((H, D, 3), bg, dtype=np.float32)

    for y in range(H):
        row_mask = occ_front[y, :]            # which x have pixels at z=0
        if not row_mask.any():
            continue

        # Representative color: median of bright front pixels in this row
        rep = _rep_color(col_front[y], row_mask)

        # Which Z slices have at least one surface voxel in this row?
        z_surface = surface_x[y, :, :].any(axis=0)   # (D,) — True where surface

        if not z_surface.any():
            # Fallback: use overall occupancy coverage
            z_surface = occ[y, :, :].any(axis=0)
        if not z_surface.any():
            continue

        # Fill those z slices with rep color + depth gradient
        max_z = int(z_surface.sum())
        for z in range(D):
            if z_surface[z]:
                # Gradient: slightly darker toward back for rounded look
                t_depth  = z / max(1, D - 1)          # 0 (front) → 1 (back)
                grad     = 1.0 - 0.25 * t_depth        # 1.0 → 0.75
                result[y, z, :] = rep * shade * grad

    # Right view: flip z so front stays on the correct edge
    if not is_left:
        result = result[:, ::-1, :]

    return _to_pil(result.astype(np.uint8), face_size, bg)


def _render_top_bottom(
    grid: VoxelGrid,
    is_top: bool,
    shade: float,
    face_size: int,
    bg: tuple,
) -> Image.Image:
    """
    Top / bottom face — filled-column renderer with hidden face culling.

    For each sprite column x:
      1. Find which Z slices have a surface voxel on the Y-facing side.
      2. Fill those (z, x) pixels with the column's topmost bright color.
      3. Apply depth gradient.
    """
    occ = grid.occupied
    col = grid.colors
    H, W, D = occ.shape

    if is_top:
        surface_y = _cull_surface(occ, axis=0, from_back=False)  # exposed TOP    face
    else:
        surface_y = _cull_surface(occ, axis=0, from_back=True)   # exposed BOTTOM face

    col_front = col[:, :, 0, :]   # (H, W, 3)
    occ_front = occ[:, :, 0]      # (H, W)

    result = np.full((D, W, 3), bg, dtype=np.float32)

    for x in range(W):
        col_mask = occ_front[:, x]
        if not col_mask.any():
            continue

        # Top color: topmost bright pixel in this column
        for y in range(H):
            if col_mask[y] and col_front[y, x, :].max() >= 40:
                top_col = col_front[y, x, :].astype(np.float32)
                break
        else:
            top_col = _rep_color(col_front[:, x], col_mask)

        z_surface = surface_y[:, x, :].any(axis=0)   # (D,)
        if not z_surface.any():
            z_surface = occ[:, x, :].any(axis=0)
        if not z_surface.any():
            continue

        for z in range(D):
            if z_surface[z]:
                t_depth = z / max(1, D - 1)
                grad    = 1.0 - 0.20 * t_depth
                result[z, x, :] = top_col * shade * grad

    if not is_top:
        result = result[::-1, :, :]

    return _to_pil(result.astype(np.uint8), face_size, bg)


def render_all_faces(
    grid:      VoxelGrid,
    face_size: int   = 300,
    bg:        tuple = (245, 245, 250),
) -> dict[str, Image.Image]:
    """
    Render all 6 orthographic faces with surface-only voxels.

    front / back  : ray-cast (actual sprite art visible)
    left / right  : filled-column, surface-culled, depth gradient
    top  / bottom : filled-column, surface-culled, depth gradient
    """
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
    """Return the 2-D depth map (H, W) int32 extracted from grid occupancy."""
    return grid.occupied.sum(axis=2).astype(np.int32)


def smooth_grid(grid: VoxelGrid, strength: float = 0.5, passes: int = 2) -> VoxelGrid:
    """
    Apply Gaussian neighbor smoothing to the depth map and return a new grid.

    Uses a 3×3 weighted kernel.  Cross-region neighbors (e.g. HEAD pixel next
    to TORSO pixel) contribute at 35 % weight so region boundaries are
    preserved.  Outline pixels (depth=1) are raised gently by averaging, which
    removes the sharp cliff between the border shell and the interior.
    """
    H, W    = grid.H, grid.W
    depth   = get_depth_map(grid).astype(np.float32)     # (H, W)
    kern    = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)

    for _ in range(passes):
        nxt = depth.copy()
        for y in range(H):
            for x in range(W):
                if depth[y, x] == 0:
                    continue
                ws = wt = 0.0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        ny, nx = y + ky, x + kx
                        if 0 <= ny < H and 0 <= nx < W and depth[ny, nx] > 0:
                            w   = float(kern[ky + 1, kx + 1])
                            rw  = 1.0 if grid.region_map[ny, nx] == grid.region_map[y, x] else 0.35
                            ws += depth[ny, nx] * w * rw
                            wt += w * rw
                if wt > 0:
                    sv = ws / wt
                    nxt[y, x] = max(1, min(MAX_DEPTH,
                                   round(depth[y, x] * (1 - strength) + sv * strength)))
        depth = nxt

    # Rebuild VoxelGrid with the smoothed depths (symmetric extrusion)
    new_D    = min(int(depth.max()), MAX_DEPTH) if depth.max() > 0 else 1
    new_grid = VoxelGrid(H, W, new_D)
    new_grid.region_map = grid.region_map.copy()

    ys, xs = np.where(depth > 0)
    for y, x in zip(ys, xs):
        d = min(int(depth[y, x]), new_D)

        # Read source colour from original grid's front voxel (symmetric placement)
        old_d      = int(grid.occupied[y, x, :].sum())
        old_zstart = (grid.D - old_d) // 2 if old_d > 0 else 0
        if old_d > 0 and old_zstart < grid.D:
            src = grid.colors[y, x, old_zstart]
        else:
            src = np.array([180, 140, 80], dtype=np.uint8)

        z_start = (new_D - d) // 2
        new_grid.occupied[y, x, z_start : z_start + d] = True
        new_grid.colors[y, x, z_start : z_start + d]   = src

    return new_grid


def export_grid_json(grid: VoxelGrid) -> dict:
    """
    Serialise a VoxelGrid to a JSON-safe dict for the Three.js viewer.

    Schema
    ------
    {
      "H": int, "W": int, "MAX_D": int,
      "depthMap":  [[int, ...], ...],          // 0 = empty pixel
      "colorMap":  [[[r,g,b]|null, ...], ...], // null where depthMap==0
      "regionMap": [[int|null, ...], ...],     // 0=HEAD 1=TORSO 2=LIMB
    }
    """
    H, W   = grid.H, grid.W
    depth  = get_depth_map(grid)

    depth_map  = depth.tolist()

    color_map, region_map = [], []
    for y in range(H):
        crow, rrow = [], []
        for x in range(W):
            if depth[y, x] > 0:
                c = grid.colors[y, x, 0]
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
        "depthMap":  depth_map,
        "colorMap":  color_map,
        "regionMap": region_map,
    }
