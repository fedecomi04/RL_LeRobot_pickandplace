"""Per-cube colour calibration by ORBIT — capture a cube's intrinsic RGB ratio over
many viewpoints, not a single frame.

The classifier in fastsam_seg separates cubes by their brightness-normalised RGB
RATIO (chromaticity r,g,b summing to 1), which is near-constant per colour. A single
snapshot only samples ONE viewpoint/exposure, so it can't tell us how TIGHT a colour's
ratio cluster is. This tool records ~`--secs` seconds of frames while you move the arm
(wrist camera) AROUND a single cube, then aggregates:

    chroma      (6,3)  mean ratio over all sampled frames      (used as the reference)
    chroma_std  (6,3)  per-channel std of the ratio            (cluster spread)
    radius      (6,)   p90 distance of a frame's ratio to the mean (per-colour accept radius)
    n_samples   (6,)   how many frames went into each colour

written to cube_color_calib.json. fastsam_seg loads `chroma` as before (backward
compatible); the new fields feed the radius/abs/margin classifiers tested in
test_color_algos_live.py.

    python -m final_utils.calib_cube_colors_orbit --camera_index 1 --secs 3

Put a SINGLE cube on the table, press its colour key to START a capture, then move the
arm around it for the countdown:
    0 red  1 blue  2 green  3 yellow  4 purple  5 orange
    s save   r reset all   q/Esc quit
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_utils.fastsam_seg import FastSamMasker, _CHROMA_REF  # noqa: E402
from final_utils.calib_cube_colors_live import pick_cube_segment  # noqa: E402
from deploy_utils.infer_linux import Cv2Camera  # noqa: E402

NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "cube_color_calib.json")
DET_W = 512


def aggregate(samples):
    """samples: (N,3) per-frame mean chromaticities. Returns (mean(3,), std(3,),
    radius float, n). radius = 90th-percentile distance of a frame to the mean, so
    a per-colour accept gate `dist <= radius` admits the bulk of real viewpoints."""
    s = np.asarray(samples, dtype=np.float64)
    mean = s.mean(axis=0)
    std = s.std(axis=0)
    dists = np.linalg.norm(s - mean, axis=1)
    radius = float(np.percentile(dists, 90)) if len(s) > 1 else 0.0
    return mean, std, radius, len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_index", type=int, default=1)
    ap.add_argument("--weights", default="FastSAM-x.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--disp_w", type=int, default=900)
    ap.add_argument("--secs", type=float, default=3.0, help="capture window per colour (s)")
    args = ap.parse_args()

    print(f"Loading FastSAM ({args.weights}) ...")
    masker = FastSamMasker(weights=args.weights, device=args.device, imgsz=args.imgsz, warmup=1)
    cam = Cv2Camera(args.camera_index)

    # Preload any EXISTING calibration so a partial re-orbit (e.g. just red+orange)
    # keeps the colours you don't recapture. Single-image calib has only `chroma`
    # (no std/radius); those load with std=None and get a default on save.
    refs = [None] * 6        # aggregated dict per colour: {mean,std,radius,n}
    preloaded = set()
    if os.path.exists(OUT):
        d = json.load(open(OUT))
        ch = d.get("chroma", [])
        std = d.get("chroma_std"); rad = d.get("radius"); nsm = d.get("n_samples")
        for i in range(min(6, len(ch))):
            refs[i] = {"mean": ch[i],
                       "std": std[i] if std else None,
                       "radius": float(rad[i]) if rad else 0.0,
                       "n": int(nsm[i]) if nsm else 0}
            preloaded.add(i)
        print(f"  loaded existing calib for: {[NAMES[i] for i in sorted(preloaded)]} "
              f"— re-orbit only the colours you want to refresh")
    capturing = -1           # colour index currently being captured, or -1
    cap_until = 0.0
    cap_samples = []
    print("Show ONE cube; press its colour key to START a capture, then ORBIT the arm "
          "around it. 0 red 1 blue 2 green 3 yellow 4 purple 5 orange | s save | r reset | q quit")
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
                ys, xs = np.where(m)
                cv2.putText(disp, f"chroma=({cur_chroma[0]:.2f},{cur_chroma[1]:.2f},{cur_chroma[2]:.2f})",
                            (int(xs.mean() * sx) - 80, int(ys.mean() * sy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # ── active capture: collect this frame's sample, count down ──────────
            if capturing >= 0:
                if cur_chroma is not None:
                    cap_samples.append([float(x) for x in cur_chroma])
                remain = cap_until - time.time()
                cv2.putText(disp, f"CAPTURING {NAMES[capturing]}  {remain:4.1f}s  n={len(cap_samples)}",
                            (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if remain <= 0:
                    if len(cap_samples) >= 3:
                        mean, std, radius, n = aggregate(cap_samples)
                        refs[capturing] = {"mean": mean.tolist(), "std": std.tolist(),
                                           "radius": radius, "n": n}
                        print(f"  captured {NAMES[capturing]}: mean={np.round(mean,3).tolist()} "
                              f"std={np.round(std,3).tolist()} radius={radius:.3f} n={n}")
                    else:
                        print(f"  {NAMES[capturing]}: too few samples ({len(cap_samples)}) — retry")
                    capturing, cap_samples = -1, []

            status = " ".join(f"{NAMES[i][:3]}{'OK' if refs[i] else '--'}" for i in range(6))
            cv2.putText(disp, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, "0-5 start capture | s save | r reset | q quit", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            dh = int(args.disp_w * h / w)
            cv2.imshow("cube colour ORBIT calibration", cv2.cvtColor(
                cv2.resize(disp, (args.disp_w, dh)), cv2.COLOR_RGB2BGR))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif ord('0') <= k <= ord('5'):
                if capturing >= 0:
                    print("  already capturing — wait for the countdown")
                else:
                    capturing = k - ord('0')
                    cap_samples = []
                    cap_until = time.time() + args.secs
                    print(f"  capturing {NAMES[capturing]} for {args.secs:.1f}s — orbit now")
            elif k == ord('r'):
                refs, capturing, cap_samples = [None] * 6, -1, []
                print("  reset")
            elif k == ord('s'):
                if any(r is None for r in refs):
                    miss = [NAMES[i] for i, r in enumerate(refs) if r is None]
                    print(f"  NOT saved — still missing: {miss}")
                else:
                    DEFAULT_STD = [0.04, 0.04, 0.04]   # for preloaded colours with no measured std
                    out = {
                        "chroma": [r["mean"] for r in refs],        # backward-compatible reference
                        "chroma_std": [r["std"] if r["std"] is not None else DEFAULT_STD for r in refs],
                        "radius": [r["radius"] for r in refs],
                        "n_samples": [r["n"] for r in refs],
                        "names": NAMES,
                    }
                    json.dump(out, open(OUT, "w"), indent=2)
                    measured = [NAMES[i] for i, r in enumerate(refs) if r["std"] is not None]
                    print(f"  saved {OUT}  (with measured std: {measured})")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
