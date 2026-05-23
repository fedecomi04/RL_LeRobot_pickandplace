"""Compare three colour-recognition algorithms LIVE, all six cubes in view.

After orbit-calibrating (final_utils/calib_cube_colors_orbit.py writes the per-cube
mean ratio + spread + radius into cube_color_calib.json), put ALL SIX cubes on the
table and orbit the wrist camera around them. Every colourful cube segment is
classified by each algorithm and the labels are drawn on it so you can see, frame by
frame, which algorithm names all six correctly with no cross-talk:

    nearest  baseline — closest of the 6 references always wins (current deploy)
    radius   closest wins only within that colour's own calibrated accept radius
    abs      closest wins only if its absolute distance <= --abs_thresh
    margin   closest wins only if it beats the runner-up by >= --margin

Per frame the HUD scores each algorithm: HIT = how many of the 6 colours were
assigned to exactly one segment (the goal: 6), DUP = colours assigned to >1 segment
(cross-talk), REJ = segments the algorithm refused to label.

    python -m final_utils.test_color_algos_live --camera_index 1

Live keys:  [ ] radius_scale -/+   ; ' abs_thresh -/+   , . margin -/+   q quit
"""
import argparse
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_utils.fastsam_seg import (  # noqa: E402
    FastSamMasker, segment_mean_chroma, classify_chroma)
from final_utils import fastsam_seg  # noqa: E402  (read _CHROMA_RADIUS after calib load)
from deploy_utils.infer_linux import Cv2Camera  # noqa: E402

NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
ABBR = ["R", "B", "G", "Y", "P", "O"]
# BGR draw colour per cube label (for the outline + text)
DRAW_BGR = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 255, 255), (200, 0, 200), (0, 140, 255)]
DET_W = 512
METHODS = ["nearest", "radius", "abs", "margin", "maha"]


def cube_segments(small, masks, min_area_frac=5e-4, max_area_frac=0.25,
                  bottom_frac=0.04, min_colorful_frac=0.30):
    """Colourful, cube-sized, non-bottom segments with their mean chromaticity.
    Returns list of (mask, chroma, (cx,cy))."""
    h, w = small.shape[:2]
    bottom = int((1.0 - bottom_frac) * h)
    out = []
    for m in masks:
        a = int(m.sum())
        if a < min_area_frac * h * w or a > max_area_frac * h * w:
            continue
        if m[bottom:, :].any():                  # bottom-anchored = gripper
            continue
        chroma, frac, n_col = segment_mean_chroma(small, m)
        if frac < min_colorful_frac:
            continue
        ys, xs = np.where(m)
        out.append((m, chroma, (float(xs.mean()), float(ys.mean()))))
    return out


def score(labels):
    """labels: list of colour index (-1 = rejected) over the segments in a frame.
    Returns (hit, dup, rej): unique colours assigned, colours assigned >1×, rejects."""
    kept = [l for l in labels if l >= 0]
    rej = sum(1 for l in labels if l < 0)
    c = Counter(kept)
    hit = sum(1 for v in c.values() if v == 1)
    dup = sum(1 for v in c.values() if v > 1)
    return hit, dup, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_index", type=int, default=1)
    ap.add_argument("--weights", default="FastSAM-x.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--disp_w", type=int, default=1100)
    ap.add_argument("--radius_scale", type=float, default=1.25)
    ap.add_argument("--abs_thresh", type=float, default=0.12)
    ap.add_argument("--margin", type=float, default=0.04)
    args = ap.parse_args()

    print(f"Loading FastSAM ({args.weights}) ...")
    masker = FastSamMasker(weights=args.weights, device=args.device, imgsz=args.imgsz, warmup=1)
    if fastsam_seg._CHROMA_RADIUS is None:
        print("WARNING: no per-colour radius in cube_color_calib.json — run "
              "calib_cube_colors_orbit first; 'radius' will fall back to abs_thresh.")
    cam = Cv2Camera(args.camera_index)
    rs, at, mg = args.radius_scale, args.abs_thresh, args.margin
    print("keys: [ ] radius_scale | ; ' abs_thresh | , . margin | q quit")
    try:
        while True:
            rgb = np.asarray(cam.async_read())
            h, w = rgb.shape[:2]
            det_h = int(round(DET_W * h / w))
            small = cv2.resize(rgb, (DET_W, det_h), interpolation=cv2.INTER_AREA)
            masks, _ = masker.segment(small)
            segs = cube_segments(small, masks)

            disp = rgb.copy()
            sx, sy = w / DET_W, h / det_h
            per_method = {m: [] for m in METHODS}      # labels this frame, per method
            for mask, chroma, (cx, cy) in segs:
                labs = {}
                for meth in METHODS:
                    lab, d = classify_chroma(chroma, method=meth, radius_scale=rs,
                                             abs_thresh=at, margin=mg)
                    labs[meth] = lab
                    per_method[meth].append(lab)
                # outline in the NEAREST colour (always defined)
                base = labs["nearest"]
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    cv2.polylines(disp, [(c.astype(np.float32) * (sx, sy)).astype(np.int32)],
                                  True, DRAW_BGR[base], 2, cv2.LINE_AA)
                # stacked per-method label (✓ shows the abbrev, · = rejected)
                px, py = int(cx * sx) - 30, int(cy * sy) - 24
                for j, meth in enumerate(["radius", "abs", "margin", "maha"]):
                    lab = labs[meth]
                    txt = f"{meth[0]}:{ABBR[lab] if lab >= 0 else '.'}"
                    col = DRAW_BGR[lab] if lab >= 0 else (160, 160, 160)
                    cv2.putText(disp, txt, (px, py + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(disp, txt, (px, py + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                col, 1, cv2.LINE_AA)

            # ── HUD: per-method score (hit/dup/rej), tunables ────────────────────
            y = 30
            cv2.putText(disp, f"segs={len(segs)}  [goal hit=6]", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for meth in METHODS:
                y += 28
                hit, dup, rej = score(per_method[meth])
                line = f"{meth:8s} hit={hit} dup={dup} rej={rej}"
                cv2.putText(disp, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(disp, f"radius_scale[ ]={rs:.2f}  abs[;']={at:.2f}  margin[,.]={mg:.3f}",
                        (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            dh = int(args.disp_w * h / w)
            cv2.imshow("colour-algo comparison (6 cubes)", cv2.cvtColor(
                cv2.resize(disp, (args.disp_w, dh)), cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('['):
                rs = round(max(0.5, rs - 0.05), 2)
            elif k == ord(']'):
                rs = round(min(3.0, rs + 0.05), 2)
            elif k == ord(';'):
                at = round(max(0.02, at - 0.01), 2)
            elif k == ord("'"):
                at = round(min(0.40, at + 0.01), 2)
            elif k == ord(','):
                mg = round(max(0.0, mg - 0.005), 3)
            elif k == ord('.'):
                mg = round(min(0.20, mg + 0.005), 3)
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"final tunables: radius_scale={rs} abs_thresh={at} margin={mg}")


if __name__ == "__main__":
    main()
