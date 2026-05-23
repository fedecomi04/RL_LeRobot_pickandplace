"""Cube masking vendored from Rsebti/robot-learning-project3 (toolset.perception).

Their approach (docs/CUBE_MASKING.md there): detect the cube by an HSV band tuned
to the BRIGHT TOP face, pick the best blob (brightest, area-bounded, not in the
bottom strip where the gripper is), fill that contour, and keep only the largest
blob — i.e. the cube's actual top-face shape, NOT a fitted rectangle. A looser
``full_blob`` mode grabs more of the (shadowed) side faces.

This is a faithful port of color_mask.py + blob_mask_utils.py + hsv_config.yaml,
adapted to take RGB (the squint deploy convention) and a goal-colour INDEX.

Source: https://github.com/Rsebti/robot-learning-project3.git
"""
from __future__ import annotations

import cv2
import numpy as np

# HSV ranges (OpenCV: H 0-179, S/V 0-255) — verbatim from their hsv_config.yaml.
HSV_RANGES = {
    "red":    [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    "orange": [((4, 120, 80), (22, 255, 255))],
    "yellow": [((12, 70, 110), (38, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((100, 120, 60), (130, 255, 255))],
    "violet": [((130, 80, 60), (170, 255, 255))],
}

# squint goal_color index -> their colour name (squint: 0 red 1 blue 2 green
# 3 yellow 4 purple 5 orange).
GOAL_IDX_TO_NAME = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "violet", 5: "orange"}


def _bright_range_from_band(lo, hi, v_floor=None, s_floor=None):
    """Tight band: brightest face (top), not sides/shadows. (their helper)."""
    vf = int(v_floor if v_floor is not None else max(int(lo[2]), 110))
    sf = int(s_floor if s_floor is not None else max(int(lo[1]), 70))
    return ((int(lo[0]), min(255, sf), min(255, vf)),
            (int(hi[0]), int(hi[1]), int(hi[2])))


def _pick_contour(contours, h, w, *, min_area, max_area_frac, seed_xy, hsv):
    """Choose the best cube contour: area-bounded, not in the bottom 18% (gripper),
    scored by brightness (or proximity to a seed point). (their helper)."""
    best_c, best_score = None, -1.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area_frac * h * w:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if cy > h * 0.82:                      # reject blobs low in the frame (gripper)
            continue
        if seed_xy is not None:
            dist = (cx - seed_xy[0]) ** 2 + (cy - seed_xy[1]) ** 2
            inside = cv2.pointPolygonTest(c, seed_xy, False) >= 0
            score = (1e6 if inside else 0) - dist
        elif hsv is not None:
            tmp = np.zeros((h, w), np.uint8)
            cv2.drawContours(tmp, [c], -1, 255, -1)
            mean_v = float(hsv[:, :, 2][tmp > 0].mean()) if np.any(tmp) else 0
            score = mean_v * 1000 + a * 0.01
        else:
            score = a
        if score > best_score:
            best_score, best_c = score, c
    if best_c is None:
        return None, 0.0
    return best_c, float(cv2.contourArea(best_c))


def _in_range_union(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
    return mask


def mask_bright_top(bgr, ranges, *, v_floor=None, exclude_bottom_frac=0.10,
                    min_area=25.0, max_area_frac=0.02, seed_xy=None):
    """Bright top-face mask: filled contour of the best blob. (their algorithm,
    extended to union all HSV ranges so red's two bands both contribute)."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    for lo, hi in ranges:
        lo_b, hi_b = _bright_range_from_band(lo, hi, v_floor=v_floor)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo_b, np.uint8), np.array(hi_b, np.uint8)))

    if exclude_bottom_frac > 0:
        mask[int(h * (1.0 - exclude_bottom_frac)):, :] = 0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, 0.0
    chosen, area = _pick_contour(contours, h, w, min_area=min_area,
                                 max_area_frac=max_area_frac, seed_xy=seed_xy, hsv=hsv)
    if chosen is None:
        return None, mask, 0.0
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [chosen], -1, 255, -1)
    M = cv2.moments(chosen)
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])), out, area


def _keep_largest_blob(mask):
    if mask is None or not np.any(mask):
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    main = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [main], -1, 255, -1)
    return out


def mask_full_blob(bgr, ranges, *, v_floor=None, exclude_bottom_frac=0.10,
                   close_kernel=9, close_iters=2, min_area=40.0, max_area_frac=0.035):
    """Looser HSV + connected-component fill — captures more of the cube sides."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    loose = _in_range_union(hsv, ranges)
    if exclude_bottom_frac > 0:
        loose[int(h * (1.0 - exclude_bottom_frac)):, :] = 0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, k3, iterations=1)
    if close_kernel > 0 and close_iters > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, kc, iterations=close_iters)
    contours, _ = cv2.findContours(loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, loose, 0.0
    chosen, area = _pick_contour(contours, h, w, min_area=min_area,
                                 max_area_frac=max_area_frac, seed_xy=None, hsv=hsv)
    if chosen is None:
        return None, loose, 0.0
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [chosen], -1, 255, -1)
    M = cv2.moments(chosen)
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])), out, area


def detect_cube_mask_rgb(rgb, goal_idx, *, mode="bright_top", v_floor=None,
                         dilate_px=0):
    """Cube mask for the squint pipeline.

    rgb: (H,W,3) uint8 RGB. goal_idx: 0-5 (squint palette order).
    mode: 'bright_top' (default, top face) or 'full_blob' (more of the sides).
    Returns (bool_mask (H,W), centroid_xy or None, area_px).
    """
    name = GOAL_IDX_TO_NAME[int(goal_idx)]
    ranges = HSV_RANGES[name]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if mode == "full_blob":
        px, mask, area = mask_full_blob(bgr, ranges, v_floor=v_floor)
    else:
        px, mask, area = mask_bright_top(bgr, ranges, v_floor=v_floor)
    if mask is None or not np.any(mask):
        return np.zeros(rgb.shape[:2], bool), px, 0.0
    mask = _keep_largest_blob(mask)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask > 0, px, float(np.count_nonzero(mask))
