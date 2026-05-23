"""FastSAM-based goal-cube segmentation for real-robot eval1.

The pick policy was trained with ONLY the goal cube + gripper visible on a clean
table. At deploy we must reproduce that: keep the gripper and the goal cube, paint
everything else a constant grey. Until now the goal cube was found by colour
(cube_mask_rsebti); this module finds it with FastSAM instead — a class-agnostic
"segment everything" model — and then picks the segment whose interior matches the
queried colour. FastSAM gives a clean, full-cube silhouette (dark side faces and
all), and the colour test only has to choose among a handful of object proposals
rather than threshold raw pixels, so it is far more robust to shadow / exposure.

Pipeline per frame:
  1. FastSAM "everything"  -> N object masks (the cube, table, gripper, bowl, ...)
  2. score each mask by how much of it is the GOAL colour (nearest-centroid in
     (hue, sat), reusing the eval1 palette / hue_calib.json)
  3. keep = goal_cube ∪ gripper ; everything else -> constant grey
     (gripper = the dark segment(s) touching the bottom edge — picked from the
      SAME FastSAM output via gripper_mask_from_segments, no baked/FK mask needed)

Timing: every segment() call is wrapped with cuda.synchronize() so the reported
millisecond cost is the true end-to-end FastSAM time (pre + infer + post), which
the caller prints each control step.

On an RTX 5070: FastSAM-s ≈ 4 ms, FastSAM-x ≈ 14 ms per 1080p frame at imgsz=640.
"""

import json
import time
from pathlib import Path

import numpy as np
import cv2

_REPO = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = _REPO / "weights"

# OpenCV-HSV palette (0-179 hue, 0-255 sat), index = goal_color, matching
# infer_linux.GOAL_HUE_CV / GOAL_SAT_CV. Overridden by hue_calib.json if present.
#                 0 red 1 blue 2 green 3 yellow 4 purple 5 orange
GOAL_HUE_CV = [4, 112, 64, 26, 149, 8]
GOAL_SAT_CV = [249, 253, 242, 242, 235, 246]
_HUE_CALIB = _REPO / "calibration" / "hue_calib.json"
_CHROMA_CALIB = _REPO / "calibration" / "cube_color_calib.json"   # real per-cube chromaticity (calib_cube_colors_live)


def _hsv_to_chroma(hue_cv, sat, v=220):
    """OpenCV (hue,sat) -> RGB chromaticity (r,g,b summing to 1). V cancels in the
    ratio, so the measured hue+sat alone fix the reference chromaticity."""
    px = np.uint8([[[int(round(hue_cv)) % 180, int(round(sat)), v]]])
    rgb = cv2.cvtColor(px, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)
    return rgb / max(float(rgb.sum()), 1e-3)


def _load_calib():
    """Set the colour references. Priority:
    1. cube_color_calib.json — REAL per-cube chromaticity measured one cube at a
       time (final_utils/calib_cube_colors_live.py). Best: matches the camera
       exactly, so red vs orange separate by their true captured ratios.
    2. hue_calib.json — measured (hue,sat); chromaticity derived from it.
    Otherwise the sim-palette default _CHROMA_REF stays.
    """
    global GOAL_HUE_CV, GOAL_SAT_CV, _CHROMA_REF, _CHROMA_STD, _CHROMA_RADIUS
    if _CHROMA_CALIB.exists():
        d = json.loads(_CHROMA_CALIB.read_text())
        ref = np.array(d["chroma"], dtype=np.float32)        # (6,3)
        ref = ref / np.clip(ref.sum(axis=1, keepdims=True), 1e-3, None)
        _CHROMA_REF = ref
        # Optional cluster-spread fields from the ORBIT calibration (per-colour
        # std + accept radius); None when only single-shot calib is present.
        _CHROMA_STD = np.array(d["chroma_std"], np.float32) if "chroma_std" in d else None
        _CHROMA_RADIUS = np.array(d["radius"], np.float32) if "radius" in d else None
        return True
    if _HUE_CALIB.exists():
        d = json.loads(_HUE_CALIB.read_text())
        GOAL_HUE_CV = [int(round(h)) for h in d["hues"]]
        if "sat" in d:
            GOAL_SAT_CV = [int(round(s)) for s in d["sat"]]
        _CHROMA_REF = np.array([_hsv_to_chroma(GOAL_HUE_CV[i], GOAL_SAT_CV[i])
                                for i in range(6)], dtype=np.float32)
        return True
    return False


# A pixel counts as a coloured-cube pixel only if it's saturated AND bright enough,
# so the desaturated table and the near-black gripper are excluded from colour scoring.
CUBE_SAT_MIN = 80
CUBE_VAL_MIN = 35
COLOR_DIST_TOL = 30      # max (hue,sat) distance to the goal centroid to count a pixel as goal
SAT_DIST_W = 1.0         # weight on saturation vs hue in the colour classifier

# ── RGB-chromaticity palette (the robust classifier) ──────────────────────────
# HSV hue collapses red/orange/yellow (hues 4/8/26) together, so they get mixed
# up. In RGB the cubes are cleanly separated by channel RATIOS: brightness-
# normalized chromaticity r=R/(R+G+B), g=G/(R+G+B) gives red r≈0.95, orange
# r≈0.73, yellow r≈0.50 — wide margins, brightness-invariant. We classify each
# FastSAM SEGMENT by the nearest reference chromaticity of its mean colour.
#                       0 red       1 blue      2 green     3 yellow    4 purple    5 orange
RGB_PALETTE = np.array([[238, 12, 0], [0, 37, 166], [4, 100, 10],
                        [255, 235, 18], [81, 39, 89], [255, 69, 27]], dtype=np.float32)
_CHROMA_REF = RGB_PALETTE / RGB_PALETTE.sum(axis=1, keepdims=True)   # (6,3) reference chromaticities
_CHROMA_STD = None        # (6,3) per-channel ratio spread, from ORBIT calib (else None)
_CHROMA_RADIUS = None     # (6,)  per-colour accept radius, from ORBIT calib (else None)
COLOR_MIN_SPREAD = 30    # only average over COLOURFUL pixels (RGB max-min ≥ this), so washed-out
                         # / near-grey / near-black pixels don't pull the mean toward neutral
CHROMA_MARGIN = 0.04     # the goal must win by this much over the runner-up colour (else ambiguous)
CHROMA_MAX_DIST = 0.20   # a segment is a CUBE only if its chromaticity is within this of some
                         # palette colour — rejects the desaturated table / achromatic gripper.
                         # 0.20 admits the darker cubes (green ~0.14, purple ~0.15 measured); the
                         # near-neutral table/bowl is kept out by the CUBE_MIN_COLORFUL_FRAC gate,
                         # not this distance, so loosening here is safe.
CUBE_MIN_COLORFUL_PX = 10    # ...and has at least this many colourful pixels to classify on
CUBE_MIN_COLORFUL_FRAC = 0.35  # ...and is MOSTLY colourful: a cube is saturated through-and-through,
                               # whereas the white/grey BOWL is only colourful at a few edge pixels, so
                               # this fraction (colourful px / segment area) keeps the bowl from ever
                               # being mistaken for a cube and kept.

# ── GOAL detection is deliberately LOOSER than the cube/distractor gates above ──
# For the goal we already know which colour we want, so the strict margin/fraction
# gates (which exist only to keep the BOWL, GRIPPER and DISTRACTORS out) are
# needlessly harsh and cause the goal cube to be missed under bad lighting. We
# require only that the goal colour be a near match and among the closest palette
# colours — no ambiguity-margin, and a much lower colourful-fraction.
GOAL_CHROMA_MAX_DIST = 0.30    # the goal colour must be within this of the segment's chroma
GOAL_MIN_COLORFUL_FRAC = 0.12  # the goal segment must be at least this colourful (vs 0.35 for cubes)
GOAL_TOPK = 3                  # ...and the goal colour must rank among the K nearest palette colours


CHROMA_CORE_FRAC = 0.5   # read colour from only the INNERMOST 50% of the segment
                         # (the pixels farthest from the border, via a distance
                         # transform), discarding the anti-aliased boundary halo,
                         # specular edges and partial table pixels that contaminate
                         # the colour and inflate a cube's ratio spread.


def _core_pixels(rgb, mask, core_frac=CHROMA_CORE_FRAC):
    """RGB of the segment's INNER CORE: keep only pixels whose distance to the mask
    border is in the top `core_frac` (e.g. 0.5 → innermost half). A per-segment
    distance transform on the mask's bounding box (cheap) gives the distance; we
    threshold at the (1-core_frac) quantile. Falls back to the full mask for tiny
    segments. Returns (px (M,3) float32, full_area)."""
    ys, xs = np.where(mask)
    full_area = len(xs)
    if full_area == 0:
        return np.zeros((0, 3), np.float32), 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if full_area < 16:                                   # too small to erode meaningfully
        return rgb[mask].astype(np.float32), full_area
    # Pad with a 1-px zero border so the distance transform always has background to
    # measure from (a mask that fills its bbox would otherwise have no interior zero).
    sub = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
    sub[1:-1, 1:-1] = mask[y0:y1, x0:x1]
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
    subb = sub.astype(bool)
    thr = float(np.quantile(dist[subb], 1.0 - core_frac))   # keep the farthest core_frac
    core = (subb & (dist >= thr))[1:-1, 1:-1]            # strip the 1-px pad → bbox coords
    if int(core.sum()) < 8:                              # degenerate (thin) mask → whole mask
        return rgb[mask].astype(np.float32), full_area
    return rgb[y0:y1, x0:x1][core].astype(np.float32), full_area


def segment_mean_chroma(rgb, mask):
    """Mean RGB chromaticity (r,g,b summing to 1) of a segment's INNER CORE
    (CHROMA_CORE_FRAC of the pixels farthest from the border), averaged over the
    COLOURFUL core pixels only. Returns (mean_chroma (3,), colourful_fraction, n_col)."""
    px, _full_area = _core_pixels(rgb, mask)
    if len(px) == 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], np.float32), 0.0, 0
    spread = px.max(axis=1) - px.min(axis=1)
    keep = (spread >= COLOR_MIN_SPREAD) & (px.sum(axis=1) > 30)
    use = px[keep] if keep.sum() >= 8 else px
    chroma = use / np.clip(use.sum(axis=1, keepdims=True), 1e-3, None)
    return chroma.mean(axis=0), float(keep.sum()) / max(1, len(px)), int(keep.sum())


# ── Mahalanobis (variance-weighted) labelling ─────────────────────────────────
# When per-colour std is calibrated (the ORBIT calib), classify by Mahalanobis
# distance — each channel's error divided by that colour's std — instead of plain
# Euclidean. This leans on the channel that actually SEPARATES near-ratio colours
# (red vs orange split cleanly on the GREEN channel, ~4σ) instead of letting a
# noisy channel (red's r-ratio) dilute the decision. std is CLAMPED to
# [FLOOR, CEIL] so a colour measured with huge spread (e.g. green) can't become a
# 'sink' that pulls every segment toward it. Distances are then in σ units, so the
# gate thresholds switch to the MAHA_* values below when this path is active.
MAHA_STD_FLOOR = 0.02
MAHA_STD_CEIL = 0.06
MAHA_CUBE_MAX = 6.0      # cube-ness: nearest-colour σ-distance must be within this
MAHA_GOAL_MAX = 7.0      # goal closeness in σ
MAHA_MARGIN = 1.5        # strict-cube: nearest must beat runner-up by this many σ
MAHA_GOAL_TOPK = 1       # with clean separation the goal must be the NEAREST colour


def _maha_active():
    return _CHROMA_STD is not None


def _label_distances(mean_ch):
    """Distance from a chromaticity to each of the 6 references in the ACTIVE
    metric: Mahalanobis (σ units, clamped std) when calibrated, else Euclidean."""
    if _CHROMA_STD is None:
        return np.linalg.norm(_CHROMA_REF - mean_ch, axis=1)
    std = np.clip(_CHROMA_STD, MAHA_STD_FLOOR, MAHA_STD_CEIL)
    z = (_CHROMA_REF - np.asarray(mean_ch, np.float32)) / std
    return np.sqrt((z ** 2).sum(axis=1))


def _cube_max_dist():
    return MAHA_CUBE_MAX if _maha_active() else CHROMA_MAX_DIST


def _goal_max_dist():
    return MAHA_GOAL_MAX if _maha_active() else GOAL_CHROMA_MAX_DIST


def _cube_margin():
    return MAHA_MARGIN if _maha_active() else CHROMA_MARGIN


def _goal_topk():
    return MAHA_GOAL_TOPK if _maha_active() else GOAL_TOPK


def classify_segment_color(rgb, mask):
    """Classify a segment by the nearest reference RGB chromaticity of its mean
    colour, in the active metric (Mahalanobis if std calibrated, else Euclidean).
    Returns (label, dist_to_each (6,), n_colourful_px)."""
    mean_ch, _frac, n_col = segment_mean_chroma(rgb, mask)
    d = _label_distances(mean_ch)
    return int(d.argmin()), d, n_col


def classify_chroma(chroma, method="nearest", radius=None, radius_scale=1.25,
                    abs_thresh=0.12, margin=0.04, ref=None):
    """Label a single chromaticity (r,g,b) by one of the comparison algorithms.

    Returns (label, dist (6,)) where label is the colour index, or -1 if the
    method rejects it (not confidently any cube colour).

      "nearest" — current deploy behaviour: the closest of the 6 references wins,
                  always. No rejection.
      "radius"  — closest wins ONLY if its distance is within that colour's own
                  measured accept radius (radius_scale * orbit-calib radius). Tight
                  colours gate strictly, fuzzy ones leniently. Falls back to
                  abs_thresh when no per-colour radius is calibrated.
      "abs"     — closest wins only if its absolute distance ≤ abs_thresh, so the
                  bowl/gripper/shadow (merely 'least wrong') is rejected.
      "margin"  — closest wins only if it beats the runner-up by ≥ margin, so
                  ambiguous in-between ratios (e.g. red↔orange) are rejected.
      "maha"    — DEPLOY method: variance-weighted (Mahalanobis) distance using the
                  calibrated per-colour std (clamped); nearest wins only if within
                  MAHA_CUBE_MAX σ and beating the runner-up by MAHA_MARGIN σ.
                  Falls back to Euclidean 'margin' when no std is calibrated.
    """
    chroma = np.asarray(chroma, np.float32)
    if method == "maha":
        d = _label_distances(chroma)        # σ units when std calibrated, else Euclidean
        order = np.argsort(d)
        best = int(order[0])
        ok = d[best] <= _cube_max_dist() and (d[order[1]] - d[best]) >= _cube_margin()
        return (best if ok else -1), d
    ref = _CHROMA_REF if ref is None else ref
    d = np.linalg.norm(ref - chroma, axis=1)
    order = np.argsort(d)
    best = int(order[0])
    if method == "nearest":
        return best, d
    if method == "radius":
        rad = radius if radius is not None else _CHROMA_RADIUS
        thr = float(rad[best]) * radius_scale if rad is not None else abs_thresh
        return (best if d[best] <= thr else -1), d
    if method == "abs":
        return (best if d[best] <= abs_thresh else -1), d
    if method == "margin":
        return (best if (d[order[1]] - d[best]) >= margin else -1), d
    raise ValueError(f"unknown method {method!r}")


def nearest_palette(rgb, hsv=None):
    """Per-pixel nearest palette colour in (hue, sat). Returns (nearest, dstack,
    cubeish): the argmin label map, the per-colour distance stack, and a mask of
    pixels saturated+bright enough to be cube material."""
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    Hi, Si = H.astype(np.float32), S.astype(np.float32)
    sat_scale = 180.0 / 255.0
    hue_d = np.stack(
        [np.minimum(np.abs(Hi - hc), 180 - np.abs(Hi - hc)) for hc in GOAL_HUE_CV], axis=0)
    sat_d = np.stack([np.abs(Si - sc) for sc in GOAL_SAT_CV], axis=0) * sat_scale
    dstack = np.sqrt(hue_d ** 2 + (SAT_DIST_W * sat_d) ** 2)
    nearest = dstack.argmin(axis=0)
    cubeish = (S >= CUBE_SAT_MIN) & (V >= CUBE_VAL_MIN)
    return nearest, dstack, cubeish


def goal_color_pixels(rgb, goal_color, hsv=None):
    """Boolean (H,W) map: pixels whose nearest palette colour is the goal colour
    (and which are saturated + bright enough to be cube material)."""
    nearest, dstack, cubeish = nearest_palette(rgb, hsv=hsv)
    gi = int(goal_color)
    return cubeish & (nearest == gi) & (dstack[gi] <= COLOR_DIST_TOL)


class FastSamMasker:
    """Loads a FastSAM model once and produces goal-cube / keep masks per frame."""

    def __init__(self, weights="FastSAM-s.pt", device="cuda", imgsz=640,
                 conf=0.4, iou=0.9, retina_masks=True, warmup=3,
                 hold_on_miss=True, max_hold_frames=120):
        from ultralytics import FastSAM
        import torch
        self._torch = torch
        w = Path(weights)
        if not w.is_absolute() and not w.exists():
            w = WEIGHTS_DIR / w.name          # prefer the repo weights/ copy
        self.model = FastSAM(str(w))
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.retina_masks = bool(retina_masks)
        # When a frame misses the cube, reuse the last good cube mask rather than
        # blanking the cube to grey (a sudden all-grey frame would jolt the policy).
        # The held mask is only reused for up to max_hold_frames consecutive misses,
        # so a cube that genuinely left the view doesn't persist forever.
        self.hold_on_miss = bool(hold_on_miss)
        self.max_hold_frames = int(max_hold_frames)
        self._last_cube = None                # last good cube mask, at det res
        self._miss_streak = 0
        _load_calib()
        for _ in range(max(0, warmup)):       # JIT/cudnn warmup so step 1 isn't slow
            dummy = np.zeros((720, 1280, 3), np.uint8)
            self._predict(dummy)

    def reset(self):
        """Forget the held cube mask (call at the start of a new episode)."""
        self._last_cube = None
        self._miss_streak = 0

    def _predict(self, rgb):
        return self.model.predict(
            rgb, device=self.device, imgsz=self.imgsz, retina_masks=self.retina_masks,
            conf=self.conf, iou=self.iou, verbose=False)

    def segment(self, rgb):
        """Run FastSAM 'everything'. Returns (masks (N,H,W) bool, sam_ms).

        sam_ms is the true end-to-end time incl. cuda sync, so it can be printed
        as the control-step cost of segmentation.
        """
        h, w = rgb.shape[:2]
        if self.device != "cpu":
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = self._predict(rgb)[0]
        if self.device != "cpu":
            self._torch.cuda.synchronize()
        sam_ms = (time.perf_counter() - t0) * 1000.0

        if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
            return np.zeros((0, h, w), bool), sam_ms
        m = res.masks.data.detach().cpu().numpy() > 0.5
        if m.shape[1:] != (h, w):              # resize masks to full frame if needed
            m = np.stack([
                cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                for mm in m])
        return m, sam_ms

    def classify_cube_segments(self, rgb, masks=None, min_area_frac=2e-4,
                               max_area_frac=0.20, bottom_frac=0.04):
        """Classify every segment by RGB chromaticity. Returns a list of
        {mask, label, area, cx, cy, dist, bottom} for the segments that are
        CONFIDENT coloured cubes — chromaticity within CHROMA_MAX_DIST of a palette
        colour, winning the runner-up by CHROMA_MARGIN, enough colourful pixels, a
        cube-plausible area. `bottom` flags segments touching the bottom edge: the
        bottom-anchored GRIPPER can take a warm colour cast under bad exposure, so
        callers use `bottom` to avoid treating it as a cube (cubes sit on the table,
        mid-frame; only the gripper reaches the bottom edge).
        """
        h, w = rgb.shape[:2]
        if masks is None:
            masks, _ = self.segment(rgb)
        frame_area = h * w
        bottom_row = int((1.0 - bottom_frac) * h)
        cubes = []
        for m in masks:
            area = int(m.sum())
            if area < min_area_frac * frame_area or area > max_area_frac * frame_area:
                continue
            label, dist, n_col = classify_segment_color(rgb, m)
            if label < 0 or n_col < CUBE_MIN_COLORFUL_PX:
                continue
            if n_col < CUBE_MIN_COLORFUL_FRAC * area:    # mostly-white bowl -> not a cube
                continue
            order = np.argsort(dist)
            if float(dist[order[0]]) > _cube_max_dist():          # not close to any cube colour
                continue
            if float(dist[order[1]] - dist[order[0]]) < _cube_margin():  # ambiguous between two
                continue
            ys, xs = np.where(m)
            cubes.append({"mask": m, "label": label, "area": area,
                          "cx": float(xs.mean()), "cy": float(ys.mean()), "dist": dist,
                          "bottom": bool(m[bottom_row:, :].any())})
        return cubes

    def goal_segments(self, rgb, goal_color, masks=None, min_area_frac=2e-4,
                      max_area_frac=0.20, bottom_frac=0.04):
        """LENIENT scan for segments matching the GOAL colour.

        Unlike classify_cube_segments (which gates strictly so the bowl/gripper/
        distractors stay out), this only has to find the one cube we already know
        the colour of, so it drops the ambiguity margin and uses the looser
        GOAL_* thresholds. A segment qualifies if it is cube-sized, colourful
        enough (GOAL_MIN_COLORFUL_FRAC), the goal colour is within
        GOAL_CHROMA_MAX_DIST of its chroma, AND the goal ranks in the nearest
        GOAL_TOPK palette colours. Returns the same dict shape as
        classify_cube_segments.
        """
        h, w = rgb.shape[:2]
        if masks is None:
            masks, _ = self.segment(rgb)
        gi = int(goal_color)
        frame_area = h * w
        bottom_row = int((1.0 - bottom_frac) * h)
        out = []
        for m in masks:
            area = int(m.sum())
            if area < min_area_frac * frame_area or area > max_area_frac * frame_area:
                continue
            label, dist, n_col = classify_segment_color(rgb, m)
            if n_col < CUBE_MIN_COLORFUL_PX:
                continue
            if n_col < GOAL_MIN_COLORFUL_FRAC * area:           # looser than the cube gate
                continue
            if float(dist[gi]) > _goal_max_dist():              # goal colour not a near match
                continue
            rank = int(np.argsort(dist).tolist().index(gi))    # how close the goal ranks
            if rank >= _goal_topk():
                continue
            ys, xs = np.where(m)
            out.append({"mask": m, "label": gi, "area": area,
                        "cx": float(xs.mean()), "cy": float(ys.mean()), "dist": dist,
                        "bottom": bool(m[bottom_row:, :].any())})
        return out

    def goal_cube_mask(self, rgb, goal_color, masks=None, hsv=None,
                       min_area_frac=2e-4, max_area_frac=0.20,
                       min_goal_frac=0.25, dilate_px=3, cubes=None):
        """Largest segment matching the goal colour (LENIENT goal_segments scan).

        Returns (bool (H,W) cube mask, dbg=(cx,cy,area,confidence) or None). Pass a
        precomputed `cubes` list (classify_cube_segments) to seed the search; any
        goal-coloured entry there is reused, but we ALSO run the lenient
        goal_segments scan so a goal cube the strict gate rejected is still found.
        """
        h, w = rgb.shape[:2]
        gi = int(goal_color)
        # Lenient goal scan over the raw segments (finds cubes the strict gate drops).
        goal = self.goal_segments(rgb, gi, masks=masks,
                                  min_area_frac=min_area_frac, max_area_frac=max_area_frac)
        if not goal and cubes is not None:                      # fall back to the strict list
            goal = [c for c in cubes if c["label"] == gi]
        if not goal:
            return np.zeros((h, w), bool), None
        # Prefer a table cube (not bottom-anchored) over a cube+gripper merge — the
        # bottom-anchored blob is the warm-cast gripper fused with the cube.
        table_goal = [c for c in goal if not c["bottom"]]
        best = max(table_goal or goal, key=lambda c: c["area"])
        cube = best["mask"]
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            cube = cv2.dilate(cube.astype(np.uint8) * 255, k) > 0
        conf = 1.0 / (1.0 + float(best["dist"][gi]))    # metric-agnostic (Euclidean or σ)
        dbg = (best["cx"], best["cy"], best["area"], conf)
        return cube, dbg

    def distractor_mask(self, rgb, goal_color, masks=None, cubes=None, dilate_px=3,
                        min_area_frac=2e-4, max_area_frac=0.20):
        """Union of every segment classified as a NON-goal cube colour — the
        distractor cubes to hide. Returns (bool (H,W), [labels])."""
        h, w = rgb.shape[:2]
        if cubes is None:
            cubes = self.classify_cube_segments(rgb, masks, min_area_frac, max_area_frac)
        gi = int(goal_color)
        out = np.zeros((h, w), bool)
        labels = []
        for c in cubes:
            # Hide a NON-goal cube colour only if it's a table cube (mid-frame), not
            # the bottom-anchored gripper (which can take a warm colour cast).
            if c["label"] != gi and not c["bottom"]:
                out |= c["mask"]
                labels.append(c["label"])
        if dilate_px > 0 and out.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            out = cv2.dilate(out.astype(np.uint8) * 255, k) > 0
        return out, labels

    def gripper_mask_from_segments(self, rgb, masks=None, hsv=None,
                                   v_max=95, spread_max=50, bottom_frac=0.04, bridge_px=8,
                                   min_area_frac=5e-4, max_area_frac=0.6, dilate_px=4):
        """Pick the gripper out of the FastSAM segments — no baked/FK mask needed.

        The SO-101 gripper is (a) DARK (low mean V), (b) ACHROMATIC (near-black:
        small mean RGB channel spread max−min), is NOT the huge bright background,
        and is a single rigid body whose base always reaches the BOTTOM EDGE of the
        wrist frame. We therefore:
          1. union the DARK + ACHROMATIC, area-bounded segments  (body + each
             finger; a finger split into its own SAM segment is included here);
          2. bridge small gaps (bridge_px) and take connected components;
          3. keep every component that reaches the bottom edge — so a finger
             returned as a SEPARATE segment is still kept as long as it's connected
             to the bottom-anchored gripper body; a stray dark blob NOT touching the
             gripper is dropped.

        The ACHROMATIC test (channel spread) is what separates the black gripper
        from COLOURED cubes — including dark/shadowed ones: a blue/purple cube has a
        large RGB spread even when dark, the gripper does not. (HSV saturation can't
        do this: near-black gripper pixels report high, noisy saturation.)
        """
        h, w = rgb.shape[:2]
        if masks is None:
            masks, _ = self.segment(rgb)
        if hsv is None:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        V = hsv[:, :, 2]
        rgb_i = rgb.astype(np.int16)
        spread = rgb_i.max(axis=2) - rgb_i.min(axis=2)   # absolute RGB chroma spread

        dark = np.zeros((h, w), bool)
        for m in masks:
            area = int(m.sum())
            if area < min_area_frac * h * w or area > max_area_frac * h * w:
                continue
            if float(V[m].mean()) <= v_max and float(spread[m].mean()) <= spread_max:
                dark |= m                                # dark AND achromatic = gripper material
        if not dark.any():
            return dark

        # Bridge small gaps between separately-segmented fingers / body, then keep
        # the connected dark blobs that reach the bottom edge.
        bridged = dark.astype(np.uint8) * 255
        if bridge_px > 0:
            kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge_px + 1, 2 * bridge_px + 1))
            bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, kb)
        n, lab = cv2.connectedComponents(bridged, 8)
        bottom = int((1.0 - bottom_frac) * h)
        keep_labels = np.unique(lab[bottom:, :])
        keep_labels = keep_labels[keep_labels != 0]
        out = np.isin(lab, keep_labels) & dark      # bottom-anchored dark chains (real dark px)
        if dilate_px > 0 and out.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            out = cv2.dilate(out.astype(np.uint8) * 255, k) > 0
        return out

    def keep_mask_frame(self, rgb, goal_color, gripper_mask=None, gripper_from_sam=False,
                        hide_distractors=True, grey=128, det_w=512, return_debug=False):
        """keep = goal_cube ∪ gripper ; everything else -> constant grey ; then any
        DISTRACTOR cube (a segment classified as a non-goal cube colour) is greyed
        ON TOP, so a distractor is hidden even where it overlaps the gripper region.

        All segmentation + colour work is done at a small DETECTION resolution
        (det_w wide); the returned masked frame is at detection resolution.

        gripper_from_sam: pick the gripper from the SAME FastSAM segments (dark +
        bottom-anchored), any jaw angle, no baked/FK mask. Unioned with gripper_mask.
        hide_distractors: classify every segment and grey those whose colour is a
        non-goal cube colour (the robust RGB-chromaticity classifier).
        grey: int 0-255 or (r,g,b). Returns (masked_rgb_detres, sam_ms[, dbg]).
        """
        h, w = rgb.shape[:2]
        if det_w and det_w < w:
            det_h = int(round(det_w * h / w))
            small = cv2.resize(rgb, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            small = rgb
            det_w, det_h = w, h
        masks, sam_ms = self.segment(small)
        # Classify every segment by RGB chromaticity ONCE; goal + distractors share it.
        cubes = self.classify_cube_segments(small, masks=masks)
        cube, cube_dbg = self.goal_cube_mask(small, goal_color, masks=masks, cubes=cubes)
        # ── Hold the last good cube mask through a missed detection ───────────────
        held = False
        if cube_dbg is not None:
            self._last_cube = cube                       # fresh detection -> remember it
            self._miss_streak = 0
        elif (self.hold_on_miss and self._last_cube is not None
              and self._last_cube.shape == cube.shape
              and self._miss_streak < self.max_hold_frames):
            cube = self._last_cube                       # miss -> reuse the previous frame's cube
            self._miss_streak += 1
            held = True
        else:
            self._miss_streak += 1                       # miss with nothing (good) to fall back on
        keep = cube.copy()
        grip = np.zeros((det_h, det_w), bool)
        if gripper_from_sam:
            grip = self.gripper_mask_from_segments(small, masks=masks)
        if gripper_mask is not None:
            g = gripper_mask
            if g.shape != (det_h, det_w):
                g = cv2.resize(g.astype(np.uint8), (det_w, det_h), interpolation=cv2.INTER_NEAREST) > 0
            grip = grip | g
        keep |= grip

        hide = np.zeros((det_h, det_w), bool)
        hide_labels = []
        if hide_distractors:
            hide, hide_labels = self.distractor_mask(small, goal_color, cubes=cubes)
            hide &= ~cube      # never hide the goal cube, even if the strict gate mislabels it

        fill = np.array([grey, grey, grey], np.float64) if np.isscalar(grey) else np.asarray(grey, np.float64)
        out = small.copy()
        out[~keep] = fill.astype(small.dtype)
        out[hide] = fill.astype(small.dtype)             # distractors greyed ON TOP of everything
        if return_debug:
            # Per-segment colour readout for the live viewer: nearest colour + its
            # distance + colourful-pixel fraction, for EVERY non-tiny segment
            # (including ones that fail the cube gate), so colour misreads are visible.
            seg_info = []
            for m in masks:
                a = int(m.sum())
                if a < 2e-4 * det_h * det_w:
                    continue
                label, dist, n_col = classify_segment_color(small, m)
                ys, xs = np.where(m)
                seg_info.append((float(xs.mean()), float(ys.mean()), int(label),
                                 float(dist.min()), n_col / max(1, a)))
            return out, sam_ms, {"cube": cube, "cube_dbg": cube_dbg, "gripper": grip,
                                 "distractors": hide, "distractor_labels": hide_labels,
                                 "masks": masks, "seg_info": seg_info,
                                 "n_masks": len(masks), "held": held, "miss_streak": self._miss_streak}
        return out, sam_ms
