"""Geometric bowl mask: project the bowl (as a cylinder) into the wrist camera
from the live arm pose, with NO appearance dependence.

Why geometric, not colour: the bowl's colour overlaps the cubes and drifts with
lighting, so an HSV mask (final_utils/mask_live.py's table mask) is unreliable
for it. The bowl pose is known instead — `bowl_xy` is taught per episode
(final_utils/teach_bowl_xy.py), its z is the (calibrated) table height, and the
camera pose is recovered live from FK on the joint angles.

Chain (all in the robot base_link frame, the frame so101_fk lives in):
    T_base_gripper = fk_frames(q)["gripper_link"]                # from joint angles
    T_gripper_cam  = (WRIST_CAMERA_BASE_POS, WRIST_CAMERA_BASE_ROT)  # fixed mount
    T_base_mount   = T_base_gripper @ T_gripper_cam              # SAPIEN cam-mount pose
    T_base_optical = T_base_mount @ MOUNT_TO_OPTICAL            # SAPIEN->OpenCV axes
Project the cylinder rim+wall points through inv(T_base_optical) + pinhole K,
take the convex hull, fill it. The hull of a projected cylinder is the bowl's
silhouette from any angle (ellipse from above, rounded rect from the side).

The mount extrinsic + fovy are copied verbatim from the sim
(envs/base_random_env.py WristCameraEnv) so sim and deploy mask identically.
Calibration error on the real arm is absorbed by the optional `nudge` (a small
pose correction) — tune it live with final_utils/tune_bowl_mask.py, then bake
the result into the constants below.

Public API (matches the call site in final_utils/mask_live.py):
    bowl_mask(q, bowl_xy, (h, w)) -> uint8 mask (0/255), 255 = bowl.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy_utils.so101_fk import fk_frames

# ── Fixed wrist-camera mount, copied from envs/base_random_env.py:1178-1185 ──
# Pose of the SAPIEN camera mount relative to gripper_link.
WRIST_CAMERA_BASE_POS = (-0.0006, 0.0498, -0.0641)                 # metres
WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-90.0),
                             np.deg2rad(91.0),
                             np.deg2rad(-35.31))                   # (roll, pitch, yaw)
WRIST_CAMERA_FOVY = np.deg2rad(76.92)                              # SAPIEN fovy (calibrated)

# ── Bowl geometry — truncated cone (frustum), measured on the real bowl ──
# Wider at the rim than the base, so the silhouette isn't a plain cylinder
# (the base ring pokes out on the far side at oblique viewing angles).
BOWL_RIM_RADIUS = 0.070       # m  (top opening ⌀ 14 cm)
BOWL_BASE_RADIUS = 0.050      # m  (bottom ⌀ 10 cm)
BOWL_HEIGHT = 0.045           # m  (rim height above the base, 4.5 cm)

# Table-z calibration (z_table(r) = a·r + b, r = radial dist from base). Loaded
# lazily from table_z_calib.json so importing this module stays light (no torch).
_TABLE_Z_CALIB_PATH = Path(__file__).resolve().parent.parent / "calibration" / "table_z_calib.json"

# OpenCV camera intrinsics (K + distortion) calibrated on the real wrist camera.
# Used in preference to the ideal fovy pinhole so the projected mask matches the
# real lens (off-centre principal point + radial/tangential distortion). Falls
# back to the fovy pinhole if the file is missing (e.g. masking sim frames).
_INTRINSICS_PATH = Path(__file__).resolve().parent.parent / "calibration" / "camera_intrinsics.json"
USE_CALIBRATED_INTRINSICS = True
_calib_cache = None


def _load_calib():
    """(K0, dist, W0, H0) from camera_intrinsics.json, or False if unavailable."""
    global _calib_cache
    if _calib_cache is None:
        try:
            d = json.loads(_INTRINSICS_PATH.read_text())
            W0, H0 = d["image_size"]
            _calib_cache = (np.array(d["K"], float), np.array(d["dist"], float),
                            float(W0), float(H0))
        except Exception:
            _calib_cache = False
    return _calib_cache

# SAPIEN camera-mount frame (x-forward, y-left, z-up) -> OpenCV optical frame
# (z-forward, x-right, y-down). Columns are the optical axes in mount coords.
MOUNT_TO_OPTICAL = np.array([
    [0.0, 0.0, 1.0],   # optical x (right) = -mount y  -> built below; see note
    [-1.0, 0.0, 0.0],  # optical y (down)  = -mount z
    [0.0, -1.0, 0.0],  # optical z (fwd)   =  mount x
])
# NOTE: MOUNT_TO_OPTICAL maps a point's OPTICAL-frame coords to MOUNT-frame
# coords (R_mount_optical). Its columns are the optical basis vectors expressed
# in the mount frame: x_opt=-y_mount, y_opt=-z_mount, z_opt=+x_mount.


def _euler_to_quat_wristcam(roll, pitch, yaw):
    """Replicates the exact euler->quat used in WristCameraEnv._update_wrist_camera_pose
    (envs/base_random_env.py:1256-1268) so our camera orientation is bit-for-bit
    the sim's. Returns (w, x, y, z)."""
    cj, sj = np.cos(pitch / 2), np.sin(pitch / 2)
    ck, sk = np.cos(yaw / 2), np.sin(yaw / 2)
    ci, si = np.cos(roll / 2), np.sin(roll / 2)
    q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk
    qw = q_py_w * ci - q_py_x * si
    qx = q_py_w * si + q_py_x * ci
    qy = q_py_y * ci + q_py_z * si
    qz = q_py_z * ci - q_py_y * si
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _quat_to_R(q):
    """(w,x,y,z) Hamilton quaternion -> 3x3 rotation (SAPIEN convention)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _pose(pos, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


# Hand-eye calibration output (final_utils/calib_extrinsics.py) overrides the
# sim-tuned mount when present. Lazily loaded; set USE_MOUNT_CALIB=False to force
# the sim default (e.g. when masking sim frames).
_MOUNT_CALIB_PATH = Path(__file__).resolve().parent.parent / "calibration" / "camera_extrinsics_handeye.json"
# 2026-05-20: keeping the old visual-tuned sim mount (the hand-eye calibration
# was reverted). Set True only to re-test the hand-eye json.
USE_MOUNT_CALIB = False
_mount_cache = None


def _load_mount_calib():
    """4x4 gripper->camera mount from bowl_mask_calib.json, or None."""
    global _mount_cache
    if _mount_cache is None:
        try:
            d = json.loads(_MOUNT_CALIB_PATH.read_text())
            _mount_cache = np.array(d["T_gripper_cam_mount"], float) if "T_gripper_cam_mount" in d else False
        except Exception:
            _mount_cache = False
    return _mount_cache if _mount_cache is not False else None


def _gripper_to_cam_mount():
    """4x4 SAPIEN camera-mount pose relative to gripper_link. Uses the hand-eye
    calibration if available, else the sim-tuned default."""
    if USE_MOUNT_CALIB:
        cal = _load_mount_calib()
        if cal is not None:
            return cal
    R = _quat_to_R(_euler_to_quat_wristcam(*WRIST_CAMERA_BASE_ROT_RAD))
    return _pose(np.asarray(WRIST_CAMERA_BASE_POS, dtype=np.float64), R)


def camera_pose_in_base(q, nudge=None):
    """Live SAPIEN camera-mount pose in the base frame, from joint angles `q`
    (sim radians, FK order [pan, lift, elbow, wrist_flex, wrist_roll, gripper]).

    `nudge`, if given, is a 6-vector (dx, dy, dz, droll, dpitch, dyaw) applied as
    a small correction in the mount frame to absorb real-camera calibration error.
    Returns a 4x4 transform (mount-coords -> base-coords)."""
    T_base_gripper = fk_frames(q)["gripper_link"]
    T_mount = T_base_gripper @ _gripper_to_cam_mount()
    if nudge is not None:
        dx, dy, dz, dr, dp, dyaw = nudge
        T_corr = _pose(np.array([dx, dy, dz], dtype=np.float64),
                       _quat_to_R(_euler_to_quat_wristcam(dr, dp, dyaw)))
        T_mount = T_mount @ T_corr
    return T_mount


def intrinsics_matrix(h, w):
    """(K 3x3, dist) for image size (h, w).

    Prefers the OpenCV calibration (camera_intrinsics.json) scaled from its
    native size to (h, w) — this carries the off-centre principal point and
    lens distortion. Distortion coeffs are resolution-independent. Falls back to
    an ideal fovy pinhole (centred, no distortion) when the file is absent."""
    calib = _load_calib() if USE_CALIBRATED_INTRINSICS else False
    if calib:
        K0, dist, W0, H0 = calib
        sx, sy = w / W0, h / H0
        K = K0.copy()
        K[0, 0] *= sx; K[0, 2] *= sx          # fx, cx scale with width
        K[1, 1] *= sy; K[1, 2] *= sy          # fy, cy scale with height
        return K, dist.astype(np.float64)
    fy = (h / 2.0) / np.tan(WRIST_CAMERA_FOVY / 2.0)
    K = np.array([[fy, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64)


def intrinsics(h, w):
    """(fx, fy, cx, cy) convenience accessor (calibrated when available)."""
    K, _ = intrinsics_matrix(h, w)
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def _table_z(x, y):
    """z of the table surface at radial distance r=hypot(x,y), from
    table_z_calib.json (z = a·r + b). Falls back to 0 (flat table) if uncalibrated."""
    try:
        c = json.loads(_TABLE_Z_CALIB_PATH.read_text())
        return float(c["a"]) * float(np.hypot(x, y)) + float(c["b"])
    except Exception:
        return 0.0


def _frustum_points(cx, cy, z_floor, base_radius, rim_radius, height, n=48):
    """2 rings of `n` points: base (z_floor, base_radius) + rim (z_floor+height,
    rim_radius), base frame. Convex hull of their projection is the truncated-cone
    silhouette."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    lo = np.column_stack([cx + base_radius * c, cy + base_radius * s, np.full(n, z_floor)])
    hi = np.column_stack([cx + rim_radius * c, cy + rim_radius * s, np.full(n, z_floor + height)])
    return np.vstack([lo, hi])                       # (2n, 3)


def project_points(pts_base, q, h, w, nudge=None):
    """Project base-frame 3D points to pixel coords through the calibrated lens
    (K + distortion via cv2.projectPoints). Returns (uv, valid) where uv is
    (N, 2) float and `valid` flags points in front of the camera."""
    T_base_optical = camera_pose_in_base(q, nudge) @ _pose(np.zeros(3), MOUNT_TO_OPTICAL)
    T_world_cam = np.linalg.inv(T_base_optical)      # base -> optical-cam
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    Z = (pts_base @ R.T + t)[:, 2]                   # optical-frame depth
    valid = Z > 1e-6
    K, dist = intrinsics_matrix(h, w)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(
        np.asarray(pts_base, dtype=np.float64).reshape(-1, 1, 3),
        rvec, t.reshape(3, 1), K, dist)
    return uv.reshape(-1, 2), valid


def bowl_mask(q, bowl_xy, hw, *, z=None, rim_radius=BOWL_RIM_RADIUS,
              base_radius=BOWL_BASE_RADIUS, height=BOWL_HEIGHT,
              margin=1.0, nudge=None, return_hull=False):
    """uint8 0/255 mask (255 = bowl) of size hw=(h, w).

    q           : joint angles, sim radians, FK order.
    bowl_xy     : (x, y) bowl centre in base frame (m).
    z           : frustum base z in base frame. Default = calibrated table z at
                  the bowl, or 0 if uncalibrated.
    rim_radius  : top opening radius (m); base_radius = bottom radius (m).
                  `margin` multiplies both for a safety border.
    height      : rim height above the base (m).
    nudge       : optional (dx,dy,dz,droll,dpitch,dyaw) camera-pose correction.
    """
    h, w = int(hw[0]), int(hw[1])
    x, y = float(bowl_xy[0]), float(bowl_xy[1])
    if z is None:
        z = _table_z(x, y)
    pts = _frustum_points(x, y, z, base_radius * margin, rim_radius * margin, height)
    uv, valid = project_points(pts, q, h, w, nudge=nudge)
    mask = np.zeros((h, w), dtype=np.uint8)
    uv = uv[valid]
    if len(uv) >= 3:
        hull = cv2.convexHull(uv.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    else:
        hull = None
    if return_hull:
        return mask, (hull.reshape(-1, 2) if hull is not None else None)
    return mask
