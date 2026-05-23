"""Standalone live FastSAM segmentation test for the wrist camera.

Two modes:

  # Live window: camera -> FastSAM -> keep goal cube + gripper, grey the rest.
  # Prints the FastSAM inference time (ms) every frame.
  python -m final_utils.fastsam_live --goal_color 0

  # Live tuning of the GOAL detection thresholds (no restart) via the window keys:
  #   0-5  set goal colour            d  toggle distractor-hiding
  #   [ ]  GOAL_CHROMA_MAX_DIST -/+    r  reset the held cube mask
  #   ; '  GOAL_MIN_COLORFUL_FRAC -/+  q  quit
  #   , .  GOAL_TOPK -/+
  # The current values are shown in the HUD (bottom-left). When the goal cube is
  # reliably kept, copy the values into final_utils/fastsam_seg.py.

  # Headless probe: grab N frames, save overlays to debug_artifacts/, print timing.
  python -m final_utils.fastsam_live --goal_color 0 --probe 30 --no-window

Flags:
  --weights FastSAM-s.pt|FastSAM-x.pt   (default FastSAM-s.pt, fastest)
  --imgsz 640        FastSAM input size (lower = faster, 512 also good)
  --goal_color 0..5  red blue green yellow purple orange
  --grey 128         constant grey fill for everything that isn't kept
  --gripper baked|none
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy_utils.infer_linux import Cv2Camera                      # noqa: E402
from final_utils.fastsam_seg import FastSamMasker      # noqa: E402
from final_utils import fastsam_seg                     # noqa: E402  (mutate GOAL_* live)
from deploy_utils import cube_gripper_mask                                # noqa: E402

COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
ARTIFACTS = Path(__file__).resolve().parent.parent / "debug_artifacts"


def overlay(rgb, masked, dbg, sam_ms, total_ms, goal_color=None, hide_distractors=True):
    """Side-by-side raw|masked BGR image with the FastSAM timing burned in.
    `masked` is at detection resolution; it's upscaled to the raw frame size."""
    h, w = rgb.shape[:2]
    dh, dw = masked.shape[:2]
    raw = rgb.copy()
    if (dh, dw) != (h, w):
        masked = cv2.resize(masked, (w, h), interpolation=cv2.INTER_NEAREST)
    # Draw every FastSAM segment boundary as a thin black outline (det-res masks
    # scaled to the full frame), so the segmentation is visible live.
    if dbg is not None and dbg.get("masks") is not None and len(dbg["masks"]):
        seg = dbg["masks"]
        sh, sw = seg.shape[1:]
        sx, sy = w / sw, h / sh
        for m in seg:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                c = (c.astype(np.float32) * (sx, sy)).astype(np.int32)
                cv2.polylines(raw, [c], True, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.polylines(masked, [c], True, (0, 0, 0), 1, cv2.LINE_AA)
    # Per-segment colour readout: label each segment with the colour it classifies
    # as + (chromaticity distance, colourful fraction), so colour misreads are visible.
    if dbg is not None and dbg.get("seg_info"):
        for cx, cy, label, dmin, cfrac in dbg["seg_info"]:
            name = COLOR_NAMES[label] if 0 <= label < len(COLOR_NAMES) else "?"
            px, py = int(cx * w / dw), int(cy * h / dh)
            cv2.putText(raw, f"{name} d{dmin:.2f} c{cfrac:.2f}", (px - 40, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(raw, f"{name} d{dmin:.2f} c{cfrac:.2f}", (px - 40, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if dbg is not None and dbg.get("cube_dbg") is not None:
        cx, cy, area, gfrac = dbg["cube_dbg"]      # det-res coords -> scale to full frame
        cv2.circle(raw, (int(cx * w / dw), int(cy * h / dh)), 10, (0, 255, 0), 2)
        cv2.putText(raw, f"cube a={area} gfrac={gfrac:.2f}", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    txt = f"FastSAM {sam_ms:5.1f}ms  loop {total_ms:5.1f}ms  ({1000.0/max(total_ms,1e-3):4.1f} Hz)"
    nM = "" if dbg is None else f"  masks={dbg.get('n_masks','?')}"
    cv2.putText(raw, txt + nM, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # ── live-tuning HUD (bottom-left of the raw panel) ────────────────────────
    gname = COLOR_NAMES[goal_color] if goal_color is not None else "?"
    hud = [f"goal[0-5]: {goal_color} {gname}",
           f"dist[ ] : {fastsam_seg.GOAL_CHROMA_MAX_DIST:.2f}",
           f"frac[;'] : {fastsam_seg.GOAL_MIN_COLORFUL_FRAC:.2f}",
           f"topk[,.] : {fastsam_seg.GOAL_TOPK}",
           f"hide[d] : {'on' if hide_distractors else 'off'}"]
    y0 = h - 14 * len(hud) - 8
    for j, line in enumerate(hud):
        y = y0 + j * 14
        cv2.putText(raw, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(raw, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    both = np.hstack([raw, masked])
    return cv2.cvtColor(both, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="FastSAM-s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--goal_color", type=int, default=0, choices=range(6))
    ap.add_argument("--grey", type=int, default=128)
    ap.add_argument("--gripper", default="sam", choices=["sam", "baked", "none"],
                    help="gripper keep: sam (dark bottom-edge segment, any jaw angle), baked, none")
    ap.add_argument("--no_hide_distractors", action="store_true",
                    help="don't grey out non-goal cube-coloured segments (default: hide them)")
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--probe", type=int, default=0, help="grab N frames, save overlays, exit")
    ap.add_argument("--no-window", action="store_true", help="don't open a cv2 window")
    args = ap.parse_args()

    print(f"Loading FastSAM ({args.weights}, imgsz={args.imgsz}) on {args.device} ...")
    masker = FastSamMasker(weights=args.weights, device=args.device, imgsz=args.imgsz,
                           conf=args.conf, iou=args.iou)
    print(f"Goal colour: {args.goal_color} ({COLOR_NAMES[args.goal_color]})")

    cam = Cv2Camera(args.camera_index)
    from_sam = args.gripper == "sam"
    if args.gripper == "baked":
        # sized lazily per-frame; load_gripper_mask caches per (w,h)
        def grip_for(w, h):
            return cube_gripper_mask.load_gripper_mask(w, h)
    else:
        def grip_for(w, h):
            return None

    show = not args.no_window
    ARTIFACTS.mkdir(exist_ok=True)
    n_target = args.probe if args.probe > 0 else 10 ** 9
    sam_times, loop_times = [], []
    goal_color = args.goal_color
    hide_distractors = not args.no_hide_distractors
    if show:
        print("Live tuning keys:  0-5 goal colour | [ ] GOAL dist -/+ | ; ' GOAL frac -/+ "
              "| , . GOAL topk -/+ | d toggle hide | r reset held | q quit")
    try:
        i = 0
        while i < n_target:
            t0 = time.perf_counter()
            rgb = np.asarray(cam.async_read())
            h, w = rgb.shape[:2]
            gm = grip_for(w, h)
            masked, sam_ms, dbg = masker.keep_mask_frame(
                rgb, goal_color, gripper_mask=gm, gripper_from_sam=from_sam,
                hide_distractors=hide_distractors,
                grey=args.grey, return_debug=True)
            total_ms = (time.perf_counter() - t0) * 1000.0
            sam_times.append(sam_ms)
            loop_times.append(total_ms)
            cd = dbg["cube_dbg"]
            if cd is not None:
                found = f"cube OK gfrac={cd[3]:.2f} area={cd[2]}"
            elif dbg.get("held"):
                found = f"MISS -> held prev ({dbg['miss_streak']})"
            else:
                found = "NO CUBE"
            print(f"[{i:4d}] FastSAM {sam_ms:6.2f}ms | loop {total_ms:6.2f}ms "
                  f"({1000.0/max(total_ms,1e-3):4.1f} Hz) | masks={dbg['n_masks']:3d} | {found}")
            vis = overlay(rgb, masked, dbg, sam_ms, total_ms,
                          goal_color=goal_color, hide_distractors=hide_distractors)
            if show:
                cv2.imshow("FastSAM live  (raw | masked)", vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                elif ord("0") <= k <= ord("5"):
                    goal_color = k - ord("0")
                    masker.reset()
                    print(f"  goal colour -> {goal_color} ({COLOR_NAMES[goal_color]})")
                elif k == ord("["):
                    fastsam_seg.GOAL_CHROMA_MAX_DIST = round(max(0.05, fastsam_seg.GOAL_CHROMA_MAX_DIST - 0.01), 2)
                    print(f"  GOAL_CHROMA_MAX_DIST -> {fastsam_seg.GOAL_CHROMA_MAX_DIST}")
                elif k == ord("]"):
                    fastsam_seg.GOAL_CHROMA_MAX_DIST = round(min(0.60, fastsam_seg.GOAL_CHROMA_MAX_DIST + 0.01), 2)
                    print(f"  GOAL_CHROMA_MAX_DIST -> {fastsam_seg.GOAL_CHROMA_MAX_DIST}")
                elif k == ord(";"):
                    fastsam_seg.GOAL_MIN_COLORFUL_FRAC = round(max(0.0, fastsam_seg.GOAL_MIN_COLORFUL_FRAC - 0.01), 2)
                    print(f"  GOAL_MIN_COLORFUL_FRAC -> {fastsam_seg.GOAL_MIN_COLORFUL_FRAC}")
                elif k == ord("'"):
                    fastsam_seg.GOAL_MIN_COLORFUL_FRAC = round(min(0.90, fastsam_seg.GOAL_MIN_COLORFUL_FRAC + 0.01), 2)
                    print(f"  GOAL_MIN_COLORFUL_FRAC -> {fastsam_seg.GOAL_MIN_COLORFUL_FRAC}")
                elif k == ord(","):
                    fastsam_seg.GOAL_TOPK = max(1, fastsam_seg.GOAL_TOPK - 1)
                    print(f"  GOAL_TOPK -> {fastsam_seg.GOAL_TOPK}")
                elif k == ord("."):
                    fastsam_seg.GOAL_TOPK = min(6, fastsam_seg.GOAL_TOPK + 1)
                    print(f"  GOAL_TOPK -> {fastsam_seg.GOAL_TOPK}")
                elif k == ord("d"):
                    hide_distractors = not hide_distractors
                    print(f"  hide_distractors -> {hide_distractors}")
                elif k == ord("r"):
                    masker.reset()
                    print("  held cube mask reset")
            elif args.probe and (i < 3 or i == n_target - 1):
                p = ARTIFACTS / f"fastsam_probe_{COLOR_NAMES[args.goal_color]}_{i:03d}.png"
                cv2.imwrite(str(p), vis)
                print(f"        saved {p}")
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if show:
            cv2.destroyAllWindows()
    if sam_times:
        s = np.array(sam_times[1:] or sam_times)
        l = np.array(loop_times[1:] or loop_times)
        print(f"\nFastSAM ms: median={np.median(s):.2f} mean={s.mean():.2f} "
              f"p95={np.percentile(s,95):.2f} | loop median={np.median(l):.2f}ms "
              f"({1000.0/np.median(l):.1f} Hz)")


if __name__ == "__main__":
    main()
