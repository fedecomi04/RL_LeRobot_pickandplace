"""Per-cube colour calibration — show ONE cube at a time, capture its REAL colour.

The RGB-chromaticity classifier in fastsam_seg separates the cubes by their colour
RATIOS, but only as well as its reference colours match the camera. Hue-derived
references mis-read red as orange under warm light. This tool measures each cube's
true chromaticity straight from the live wrist camera — one cube in view at a time,
so there's zero ambiguity — and writes cube_color_calib.json, which fastsam_seg
loads as the references (taking priority over hue_calib.json).

    python -m final_utils.calib_cube_colors_live --camera_index 1

In the window, put a SINGLE cube on the table (gripper may be in view), then press
the key for its colour to capture it:
    0 red   1 blue   2 green   3 yellow   4 purple   5 orange
    s  save cube_color_calib.json      r  reset all      q/Esc  quit

The detected cube is outlined; its live chromaticity and current nearest colour are
shown. Capture all six, press s. Re-running overwrites.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_utils.fastsam_seg import (  # noqa: E402
    FastSamMasker, segment_mean_chroma, _CHROMA_REF)
from deploy_utils.infer_linux import Cv2Camera  # noqa: E402

NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "cube_color_calib.json")
DET_W = 512


def pick_cube_segment(small, masks, min_frac=0.40):
    """The single cube in view: the most-colourful, non-bottom segment (skips the
    bottom-anchored gripper and the desaturated table). Returns (mask, chroma) or None."""
    h, w = small.shape[:2]
    bottom = int(0.96 * h)
    best, best_n = None, 0
    for m in masks:
        a = int(m.sum())
        if a < 5e-4 * h * w or a > 0.25 * h * w:
            continue
        if m[bottom:, :].any():                 # bottom-anchored -> gripper, not a cube
            continue
        chroma, frac, n_col = segment_mean_chroma(small, m)
        if frac < min_frac:                     # must be mostly colourful
            continue
        if n_col > best_n:
            best_n, best = n_col, (m, chroma)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_index", type=int, default=1)
    ap.add_argument("--weights", default="FastSAM-x.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--disp_w", type=int, default=900)
    args = ap.parse_args()

    print(f"Loading FastSAM ({args.weights}) ...")
    masker = FastSamMasker(weights=args.weights, device=args.device, imgsz=args.imgsz, warmup=1)
    cam = Cv2Camera(args.camera_index)
    refs = [None] * 6                            # captured chromaticity per colour
    print("Show ONE cube; press its colour key to capture. "
          "0 red 1 blue 2 green 3 yellow 4 purple 5 orange | s save | r reset | q quit")
    try:
        while True:
            rgb = np.asarray(cam.async_read())
            h, w = rgb.shape[:2]
            det_h = int(round(DET_W * h / w))
            small = cv2.resize(rgb, (DET_W, det_h), interpolation=cv2.INTER_AREA)
            masks, _ = masker.segment(small)
            pick = pick_cube_segment(small, masks)

            disp = rgb.copy()
            sx, sy = w / DET_W, h / det_h
            cur_chroma = None
            if pick is not None:
                m, cur_chroma = pick
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    cv2.polylines(disp, [(c.astype(np.float32) * (sx, sy)).astype(np.int32)],
                                  True, (0, 0, 0), 2, cv2.LINE_AA)
                # nearest captured colour (else nearest current reference)
                ref = np.array([r if r is not None else _CHROMA_REF[i] for i, r in enumerate(refs)])
                d = np.linalg.norm(ref - cur_chroma, axis=1)
                ys, xs = np.where(m)
                cv2.putText(disp, f"cube chroma=({cur_chroma[0]:.2f},{cur_chroma[1]:.2f},{cur_chroma[2]:.2f}) "
                            f"~{NAMES[int(d.argmin())]} d{d.min():.2f}",
                            (int(xs.mean() * sx) - 80, int(ys.mean() * sy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            status = " ".join(f"{NAMES[i][:3]}{'OK' if refs[i] is not None else '--'}" for i in range(6))
            cv2.putText(disp, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, "0-5 capture | s save | r reset | q quit", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            dh = int(args.disp_w * h / w)
            cv2.imshow("cube colour calibration", cv2.cvtColor(
                cv2.resize(disp, (args.disp_w, dh)), cv2.COLOR_RGB2BGR))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif ord('0') <= k <= ord('5'):
                if cur_chroma is None:
                    print(f"  [{NAMES[k - ord('0')]}] no cube detected — show one colourful cube")
                else:
                    i = k - ord('0')
                    refs[i] = [float(x) for x in cur_chroma]
                    print(f"  captured {NAMES[i]}: chroma={np.round(cur_chroma, 3).tolist()}")
            elif k == ord('r'):
                refs = [None] * 6
                print("  reset")
            elif k == ord('s'):
                if any(r is None for r in refs):
                    miss = [NAMES[i] for i, r in enumerate(refs) if r is None]
                    print(f"  NOT saved — still missing: {miss}")
                else:
                    json.dump({"chroma": refs, "names": NAMES}, open(OUT, "w"), indent=2)
                    print(f"  saved {OUT}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
