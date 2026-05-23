"""Recover the bowl's fixed world (base-frame) XY from rim-ellipse detections.

The bowl is world-fixed; only the wrist camera moves. So in any frame where we
detect the rim ellipse, the ray from the camera through the ellipse centroid
must pass through the bowl's rim centre. We know the camera pose per frame (FK
on the recorded qpos + the wrist mount extrinsic) and the rim's height above the
table, so each detection back-projects to one world-XY estimate. Aggregating
over many frames (robust median) pins the bowl down once, after which it's known
for every frame — including the ones where the rim is occluded or cut off.

This also cross-checks the extrinsics: the recovered XY should match the known
bowl_xyz in the run metadata.

Usage:
    python -m final_utils.solve_bowl_xy DEPLOY_RUN_DIR [--every 5] [--debug]
"""
import argparse
import json
import os

import cv2
import numpy as np

from final_utils import bowl_mask as bm
from final_utils.test_bowl_ellipse import find_bowl_ellipse


def solve_bowl_xy_from_frame(img, q, *, z_rim=None, min_score=0.45):
    """Recover the bowl's base-frame (x, y) from a single frame.

    Detects the rim ellipse, back-projects its whole boundary onto the rim
    plane, and returns the centroid of the world-space ring (more robust than
    the image centroid, which has a perspective bias). Returns
    (xy, score, ellipse) with xy=None if no confident detection.

    img : H×W×3 uint8 (RGB or BGR — only luma is used).
    q   : joint angles for this frame (sim radians, FK order).
    """
    if z_rim is None:
        z_rim = bm.BOWL_HEIGHT
    h, w = img.shape[:2]
    ell, score = find_bowl_ellipse(img)
    if ell is None or score < min_score:
        return None, score, None
    (cx, cy), (a1, a2), ang = ell
    poly = cv2.ellipse2Poly((int(cx), int(cy)), (int(a1 / 2), int(a2 / 2)),
                            int(ang), 0, 360, 10)
    world = []
    for u, v in poly:
        p = backproject_to_plane((float(u), float(v)), q, h, w, z_rim)
        if p is not None:
            world.append(p[:2])
    if len(world) < 5:
        return None, score, ell
    return np.mean(world, axis=0), score, ell


def backproject_to_plane(uv, q, h, w, z_plane):
    """World-frame (base) point where the camera ray through pixel uv meets the
    horizontal plane z = z_plane. Returns (3,) or None if the ray is parallel."""
    # Camera (optical-frame) pose in base coords.
    T_base_optical = bm.camera_pose_in_base(q) @ bm._pose(np.zeros(3), bm.MOUNT_TO_OPTICAL)
    C = T_base_optical[:3, 3]                       # camera centre in base
    R = T_base_optical[:3, :3]                       # optical -> base
    K, dist = bm.intrinsics_matrix(h, w)
    # Undistort pixel -> normalized optical ray [x, y, 1].
    norm = cv2.undistortPoints(np.array([[uv]], np.float64), K, dist).reshape(2)
    d_opt = np.array([norm[0], norm[1], 1.0])
    d = R @ d_opt                                    # ray direction in base
    if abs(d[2]) < 1e-9:
        return None
    s = (z_plane - C[2]) / d[2]
    if s <= 0:                                       # plane is behind the camera
        return None
    return C + s * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    meta = json.loads(open(os.path.join(args.run_dir, "metadata.json")).read())
    traj = np.load(os.path.join(args.run_dir, "trajectory.npz"))
    qpos = traj["qpos"]
    cap = cv2.VideoCapture(os.path.join(args.run_dir, meta["video"]))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    z_table = 0.0                                    # base-frame table height
    z_rim = z_table + bm.BOWL_HEIGHT                 # ellipse centroid sits at the rim

    ests = []
    used = []
    for f in range(0, min(n, len(qpos)), args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        h, w = im.shape[:2]
        ell, score = find_bowl_ellipse(im)
        if ell is None or score < args.min_score:
            continue
        (cx, cy), _, _ = ell
        p = backproject_to_plane((cx, cy), qpos[f], h, w, z_rim)
        if p is None:
            continue
        ests.append(p[:2])
        used.append(f)

    if not ests:
        print("No usable detections.")
        return

    ests = np.array(ests)
    med = np.median(ests, axis=0)
    # Robust spread: median absolute deviation -> std-equivalent.
    mad = np.median(np.abs(ests - med), axis=0) * 1.4826

    print(f"frames used        : {len(ests)} (every {args.every})")
    print(f"recovered bowl XY  : ({med[0]*100:.1f}, {med[1]*100:.1f}) cm")
    print(f"spread (robust std): ({mad[0]*100:.1f}, {mad[1]*100:.1f}) cm")
    if "bowl_xyz" in meta:
        gx, gy = meta["bowl_xyz"][0], meta["bowl_xyz"][1]
        print(f"metadata bowl XY   : ({gx*100:.1f}, {gy*100:.1f}) cm")
        print(f"error vs metadata  : ({(med[0]-gx)*100:+.1f}, {(med[1]-gy)*100:+.1f}) cm "
              f"= {np.hypot(med[0]-gx, med[1]-gy)*100:.1f} cm")

    if args.debug:
        # Project the solved frustum back onto a few frames to eyeball the fit.
        out = "/tmp/bowl_frames"
        os.makedirs(out, exist_ok=True)
        for f in used[:: max(1, len(used) // 5)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            h, w = im.shape[:2]
            _, hull = bm.bowl_mask(qpos[f], med, (h, w), return_hull=True)
            if hull is not None:
                cv2.polylines(im, [hull.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out, f"solved_f{f:04d}.png"), im)
            print("  debug ->", os.path.join(out, f"solved_f{f:04d}.png"))
    cap.release()


if __name__ == "__main__":
    main()
