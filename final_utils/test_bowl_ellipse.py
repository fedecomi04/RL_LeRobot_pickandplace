"""Quick experiment: detect the bowl rim as an ellipse on recorded frames.

The bowl is white on a white table, so color segmentation fails. But the rim
is a circle, which projects to an ellipse under any camera pose. Here we try to
find it from edges alone (no camera pose, no extrinsics).

Usage:
    python -m final_utils.test_bowl_ellipse FRAME.png [FRAME2.png ...]
    python -m final_utils.test_bowl_ellipse --video V.mp4 --every 100
Writes <name>_ellipse.png debug overlays next to each input (or to /tmp).
"""
import argparse
import os

import cv2
import numpy as np


def find_bowl_ellipse(bgr, debug=None):
    """Return (ellipse, score) for the best bowl-rim candidate, or (None, 0).

    ellipse is the cv2 tuple ((cx,cy),(MA,ma),angle).
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Boost local contrast so the faint white-on-white rim shows up.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)

    # Edges. Auto thresholds from the median.
    med = float(np.median(blur))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(blur, lo, hi)
    # Close small gaps so the rim forms a continuous contour.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    img_area = h * w
    best = None
    best_score = 0.0
    cand = []
    for c in cnts:
        if len(c) < 30:
            continue
        try:
            ell = cv2.fitEllipse(c)
        except cv2.error:
            continue
        (cx, cy), axes, ang = ell
        major = max(axes)
        minor = min(axes)
        if minor <= 1:
            continue
        # Center must be inside the frame (rejects giant off-screen fits).
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        area = np.pi * (major / 2) * (minor / 2)
        # Plausibility: a bowl occupies a meaningful but not huge slice.
        if not (0.01 * img_area < area < 0.5 * img_area):
            continue
        aspect = minor / major  # in (0, 1]
        if aspect < 0.3:  # too squashed to be a rim seen from above
            continue
        # Edge support: fraction of the fitted ellipse outline that lands on an
        # actual edge pixel. This is what separates a real rim from a random fit.
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, ell, 255, 2)
        outline = int(mask.sum() // 255)
        if outline == 0:
            continue
        hit = int(cv2.bitwise_and(mask, edges).sum() // 255)
        support = hit / outline
        if support < 0.4:  # most of the outline must sit on real edges
            continue
        score = support * (0.5 + 0.5 * aspect)
        cand.append((score, support, ell))
        if score > best_score:
            best_score = score
            best = ell

    if debug is not None:
        dbg = bgr.copy()
        for score, support, ell in sorted(cand, key=lambda x: x[0])[-8:]:
            cv2.ellipse(dbg, ell, (0, 180, 255), 1)
        if best is not None:
            cv2.ellipse(dbg, best, (0, 0, 255), 2)
            (cx, cy), _, _ = best
            cv2.circle(dbg, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        debug[...] = np.hstack([dbg, edges_bgr])

    return best, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="*")
    ap.add_argument("--video")
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--outdir", default="/tmp/bowl_frames")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    items = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(0, n, args.every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if ok:
                items.append((f"frame{f:04d}", im))
        cap.release()
    for p in args.frames:
        items.append((os.path.splitext(os.path.basename(p))[0], cv2.imread(p)))

    for name, im in items:
        if im is None:
            continue
        dbg = np.zeros((im.shape[0], im.shape[1] * 2, 3), np.uint8)
        ell, score = find_bowl_ellipse(im, debug=dbg)
        out = os.path.join(args.outdir, f"{name}_ellipse.png")
        cv2.imwrite(out, dbg)
        msg = f"{name}: score={score:.2f}"
        if ell is not None:
            (cx, cy), (MA, ma), ang = ell
            msg += f" center=({cx:.0f},{cy:.0f}) axes=({MA:.0f},{ma:.0f}) ang={ang:.0f}"
        print(msg, "->", out)


if __name__ == "__main__":
    main()
