"""Keep-only masking for the SO-101 wrist camera (real-robot inference).

The pick policy was trained with *only* the goal cube on the table, so at deploy
time we must show it a frame where everything except (a) the gripper and (b) the
goal cube is painted out. This module builds that "keep" mask from two pieces:

1. GRIPPER SILHOUETTE — the wrist camera is rigidly mounted on gripper_link, so
   the gripper's silhouette in the wrist image is CONSTANT (verified: robot-seg
   IoU = 1.0 across arm poses and the full jaw range). So we BAKE one pixel-perfect
   mask from SAPIEN's own renderer (`--bake`) and AND it into every frame — no
   per-frame rendering, FK or trimesh at deploy, just a static mask resize.
   A geometric fallback (GripperRenderer) is kept: it builds the OpenCV camera
   extrinsic from qpos via FK (verified against ManiSkill to < 1e-6) and rasterizes
   the URDF meshes with near-plane clipping + backface culling. It's used as the
   bake tool's cross-check and a sim-free fallback when no baked mask is present.

2. GOAL-CUBE REGION — the goal cube's top face is found by colour (its hue/sat),
   but only the *top* face is reliably bright; the side faces fall into shadow.
   So once the top face is found we fit its square, deduce the rest of the cube's
   footprint and expand slightly, so the whole cube (dark faces included) is kept.

keep_mask = gripper ∪ goal_cube ; everything else is painted with the table mean.
The goal-cube keep is unioned LAST, so even if the gripper hull slightly overlaps
the cube between the fingers, the cube is never painted out.

Run `python cube_gripper_mask.py --validate` to render the gripper silhouette
from a live ManiSkill env and compare it against SAPIEN's segmentation.
"""

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2

# FK joint-origin transforms, shared with so101_fk so the chain stays in sync.
from so101_fk import (
    _T, _Rz, _rpy,
    _ORIGIN_PAN, _ORIGIN_LIFT, _ORIGIN_ELBOW,
    _ORIGIN_WRISTFLEX, _ORIGIN_WRISTROLL, _ORIGIN_GRIPPER,
)

# ── Wrist-camera mount + axis convention (verified against ManiSkill) ─────────
# Mount pose relative to gripper_link (from WristCameraEnv in base_random_env.py).
WRIST_CAMERA_BASE_POS = np.array([-0.0006, 0.0498, -0.0641])
WRIST_CAMERA_BASE_RPY = (np.deg2rad(-90), np.deg2rad(91), np.deg2rad(-35.31))  # roll,pitch,yaw
WRIST_CAMERA_FOV = np.deg2rad(76.92)  # vertical FOV (SAPIEN fovy), 16:9 aspect
# SAPIEN camera-local → OpenCV camera-local axis permutation: R_cv = R_AXIS · R_mountᵀ.
# (Confirmed constant across poses; see the probes in the module history.)
R_AXIS = np.array([[0.0, -1.0, 0.0],
                   [0.0,  0.0, -1.0],
                   [1.0,  0.0,  0.0]])

MESH_DIR = Path(__file__).parent / "envs" / "robot" / "meshes"
URDF_PATH = Path(__file__).parent / "envs" / "robot" / "so101.urdf"


def _euler2quat_sim(roll, pitch, yaw):
    """Exact replica of WristCameraEnv._update_wrist_camera_pose's euler→quat,
    so R_LOCAL matches the sim mount rotation bit-for-bit (a plain Rz·Ry·Rx does
    NOT — verified)."""
    cj, sj = np.cos(pitch / 2), np.sin(pitch / 2)
    ck, sk = np.cos(yaw / 2), np.sin(yaw / 2)
    ci, si = np.cos(roll / 2), np.sin(roll / 2)
    qpw, qpx, qpy, qpz = cj * ck, sj * sk, sj * ck, cj * sk
    qw = qpw * ci - qpx * si
    qx = qpw * si + qpx * ci
    qy = qpy * ci + qpz * si
    qz = qpz * ci - qpy * si
    return np.array([qw, qx, qy, qz])


def _quat2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# Mount rotation relative to gripper_link, baked once.
R_LOCAL = _quat2R(_euler2quat_sim(*WRIST_CAMERA_BASE_RPY))


def link_transforms(q):
    """Base-frame 4×4 transform for every link the wrist camera can see.

    q: length-6 [pan, lift, elbow, wrist_flex, wrist_roll, gripper] (rad).
    The robot base is the world origin on the real robot (root = identity).
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    T_shoulder = _ORIGIN_PAN @ _Rz(q[0])
    T_upper = T_shoulder @ _ORIGIN_LIFT @ _Rz(q[1])
    T_lower = T_upper @ _ORIGIN_ELBOW @ _Rz(q[2])
    T_wrist = T_lower @ _ORIGIN_WRISTFLEX @ _Rz(q[3])
    T_gripper = T_wrist @ _ORIGIN_WRISTROLL @ _Rz(q[4])
    T_jaw = T_gripper @ _ORIGIN_GRIPPER @ _Rz(q[5])
    return {
        "base_link": np.eye(4),
        "shoulder_link": T_shoulder,
        "upper_arm_link": T_upper,
        "lower_arm_link": T_lower,
        "wrist_link": T_wrist,
        "gripper_link": T_gripper,
        "moving_jaw_so101_v1_link": T_jaw,
    }


def camera_extrinsic_cv(q):
    """3×4 OpenCV world→camera extrinsic [R | t] from qpos alone (no sim)."""
    Tg = link_transforms(q)["gripper_link"]
    Rg = Tg[:3, :3]
    R_mount = Rg @ R_LOCAL
    center = Rg @ WRIST_CAMERA_BASE_POS + Tg[:3, 3]   # camera center == mount position
    R_cv = R_AXIS @ R_mount.T
    t_cv = -R_cv @ center
    return np.hstack([R_cv, t_cv[:, None]])


def camera_intrinsic(width, height, fov=WRIST_CAMERA_FOV):
    """OpenCV pinhole K for a 16:9 frame. fx==fy from the vertical FOV (square px)."""
    fy = (height / 2.0) / np.tan(fov / 2.0)
    fx = fy
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]])


def _clip_triangle_near(tri_cam, near):
    """Sutherland-Hodgman clip a camera-space triangle (3,3) against the plane
    z = near. Returns the clipped polygon (k,3) with all z >= near, or None."""
    poly = []
    for i in range(3):
        a = tri_cam[i]
        b = tri_cam[(i + 1) % 3]
        a_in = a[2] >= near
        b_in = b[2] >= near
        if a_in:
            poly.append(a)
        if a_in != b_in:                       # edge crosses the near plane
            t = (near - a[2]) / (b[2] - a[2])
            poly.append(a + t * (b - a))
    return np.array(poly) if len(poly) >= 3 else None


# ── Robot mesh geometry (loaded once) ─────────────────────────────────────────
class GripperRenderer:
    """Projects the robot's visual meshes into the wrist camera to make a
    gripper/arm silhouette mask (sim-free fallback / bake cross-check). Meshes are
    loaded once; per-frame work is matrix multiplies + per-triangle fills with
    near-plane clipping and backface culling. NOTE: the camera is mounted <6 cm
    from the gripper, so this close-range rasterization is imperfect at the edges
    — prefer the baked SAPIEN mask (bake_gripper_mask) for deploy."""

    # Links worth drawing — everything the down-looking wrist camera can catch.
    # Ordered near→far; far arm links rarely intersect the frame but are cheap.
    DRAW_LINKS = (
        "gripper_link",
        "moving_jaw_so101_v1_link",
        "wrist_link",
        "lower_arm_link",
    )

    NEAR = 0.01   # near plane (m): triangles are clipped to z >= NEAR before projecting.
                  # The camera is mounted ON the gripper, so geometry within a few cm
                  # (and some behind the camera) must be clipped, not dropped, or the
                  # silhouette spikes / vanishes.
    DECIMATE_FACES = 2500   # per visual part, for fast per-triangle fill

    def __init__(self, urdf_path=URDF_PATH, mesh_dir=MESH_DIR, draw_links=None):
        self.mesh_dir = Path(mesh_dir)
        self.draw_links = tuple(draw_links) if draw_links else self.DRAW_LINKS
        # link name -> list of (verts (N,3) in LINK frame, faces (M,3) int)
        self.link_parts = self._load_visuals(Path(urdf_path))

    def _load_visuals(self, urdf_path):
        import trimesh
        tree = ET.parse(urdf_path)
        out = {}
        for link in tree.getroot().findall("link"):
            name = link.get("name")
            if name not in self.draw_links:
                continue
            parts = []
            for vis in link.findall("visual"):
                mesh_el = vis.find("geometry/mesh")
                if mesh_el is None:
                    continue
                fn = os.path.basename(mesh_el.get("filename"))
                mpath = self.mesh_dir / fn
                if not mpath.exists():
                    continue
                origin = vis.find("origin")
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if origin is not None:
                    if origin.get("xyz"):
                        xyz = [float(v) for v in origin.get("xyz").split()]
                    if origin.get("rpy"):
                        rpy = [float(v) for v in origin.get("rpy").split()]
                T_vis = _T(xyz, rpy)
                m = trimesh.load(mpath, force="mesh")
                if len(m.faces) > self.DECIMATE_FACES:
                    try:
                        m = m.simplify_quadric_decimation(face_count=self.DECIMATE_FACES)
                    except Exception:
                        pass
                v = np.asarray(m.vertices, dtype=np.float64)
                vh = (T_vis @ np.c_[v, np.ones(len(v))].T).T[:, :3]
                parts.append((vh, np.asarray(m.faces, dtype=np.int64)))
            if parts:
                out[name] = parts
        return out

    def render_mask(self, q, width, height, dilate_px=2):
        """Boolean (H,W) gripper/arm silhouette for the given qpos.

        Per-triangle rasterization with near-plane rejection: triangles with any
        vertex at/behind the camera are dropped (the camera is mounted ON the
        gripper, so the motor body straddles the near plane — a convex hull would
        spike to infinity). Filled triangles are unioned per part."""
        ext = camera_extrinsic_cv(q)
        R_cv, t_cv = ext[:, :3], ext[:, 3]
        K = camera_intrinsic(width, height)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Tl = link_transforms(q)
        mask = np.zeros((height, width), np.uint8)

        near = self.NEAR
        for name, parts in self.link_parts.items():
            Tlink = Tl[name]
            for vh, faces in parts:
                Pw = (Tlink @ np.c_[vh, np.ones(len(vh))].T).T[:, :3]
                Pc = Pw @ R_cv.T + t_cv          # world -> camera (OpenCV +z fwd)
                # Backface cull in camera space: keep only triangles whose outward
                # normal faces the camera. Removes far-side / behind-camera faces
                # that otherwise project to spikes (the motor housing the camera is
                # embedded in). trimesh meshes have consistent outward winding.
                v0, v1, v2 = Pc[faces[:, 0]], Pc[faces[:, 1]], Pc[faces[:, 2]]
                nrm = np.cross(v1 - v0, v2 - v0)
                facing = (nrm * v0).sum(axis=1) < 0
                faces = faces[facing]
                if len(faces) == 0:
                    continue
                zf = Pc[faces][..., 2]           # (T,3) per-triangle vertex depths
                n_front = (zf >= near).sum(axis=1)

                # Triangles fully in front: project & fill in bulk.
                full = n_front == 3
                if full.any():
                    tri_cam = Pc[faces[full]]    # (T,3,3)
                    u = fx * tri_cam[..., 0] / tri_cam[..., 2] + cx
                    v = fy * tri_cam[..., 1] / tri_cam[..., 2] + cy
                    polys = [p for p in np.stack([u, v], -1).astype(np.int32)]
                    cv2.fillPoly(mask, polys, 255)

                # Triangles straddling the near plane: clip then fill (per-tri).
                strad = np.where((n_front >= 1) & (n_front < 3))[0]
                for fi in strad:
                    poly_cam = _clip_triangle_near(Pc[faces[fi]], near)
                    if poly_cam is None:
                        continue
                    u = fx * poly_cam[:, 0] / poly_cam[:, 2] + cx
                    v = fy * poly_cam[:, 1] / poly_cam[:, 2] + cy
                    cv2.fillConvexPoly(mask, np.stack([u, v], -1).astype(np.int32), 255)

        if dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            mask = cv2.dilate(mask, k)
        return mask > 0


# ── Goal-cube mask ────────────────────────────────────────────────────────────
# Vendored from Rsebti/robot-learning-project3 (see cube_mask_rsebti.py): HSV
# bright-top detection → best blob (brightest, area-bounded, above the bottom
# gripper strip) → filled contour, largest blob only. The cube's actual top-face
# shape, NOT a fitted rectangle. 'full_blob' mode grabs more of the side faces.
import cube_mask_rsebti

CUBE_MODE = "bright_top"   # 'bright_top' (top face) or 'full_blob' (more sides)
CUBE_DILATE_PX = 3         # grow the cube blob slightly so its border is fully kept


def goal_cube_mask(rgb, goal_color, hsv=None, mode=None, dilate_px=None):
    """Boolean (H,W) mask of the goal cube, via the Rsebti bright-top/blob detector.

    Returns (bool_mask, dbg) where dbg = (cx, cy, area) or None if not found.
    `hsv` is accepted for call-compatibility but unused (the detector recomputes it).
    """
    mode = CUBE_MODE if mode is None else mode
    dilate_px = CUBE_DILATE_PX if dilate_px is None else dilate_px
    mask, px, area = cube_mask_rsebti.detect_cube_mask_rgb(
        rgb, goal_color, mode=mode, dilate_px=dilate_px)
    if px is None:
        return mask, None
    return mask, (px[0], px[1], area)


# ── Baked gripper mask (PRIMARY path) ─────────────────────────────────────────
# The wrist camera is rigidly mounted on gripper_link, so the gripper's silhouette
# in the wrist image is CONSTANT (verified: robot-seg IoU=1.0 across arm poses and
# the full jaw range). So we bake ONE pixel-perfect mask from SAPIEN's own renderer
# (correct occlusion, no near-plane spikes) and reuse it every frame — no per-frame
# FK / projection / trimesh needed at deploy. The hand-rolled GripperRenderer above
# is kept only as the bake tool's cross-check and a sim-free fallback.
GRIPPER_MASK_PATH = Path(__file__).parent.parent / "calibration" / "gripper_mask.png"
_BAKED = {}   # (w,h) -> bool mask, resized & cached


def load_gripper_mask(width, height, path=GRIPPER_MASK_PATH):
    """Load the baked gripper mask (a 16:9 PNG) and resize to (width,height).
    Returns a bool (H,W) array, or None if the baked file doesn't exist."""
    key = (width, height)
    if key in _BAKED:
        return _BAKED[key]
    p = Path(path)
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST) > 127
    _BAKED[key] = m
    return m


def bake_gripper_mask(width=1920, height=1080, dilate_px=6,
                      out_path=GRIPPER_MASK_PATH, seed=0):
    """Render the robot's wrist-camera silhouette ONCE in SAPIEN and save it as a
    boolean PNG. Resolution-independent (16:9 FOV), so deploy resizes it freely.
    dilate_px (at the bake resolution) pads for wrist-mount calibration error."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import envs.place       # noqa: F401

    env = gym.make("SO101PlaceCube-v1", num_envs=1, obs_mode="rgb+segmentation",
                   sensor_configs={"width": width, "height": height},
                   domain_randomization=False)
    env.reset(seed=seed)
    be = env.unwrapped
    be._update_wrist_camera_pose()
    if be.gpu_sim_enabled:
        be.scene._gpu_apply_all()
        be.scene.px.gpu_update_articulation_kinematics()
        be.scene._gpu_fetch_all()
    seg = be.scene.sensors["base_camera"].get_obs(
        rgb=False, segmentation=True, position=False)["segmentation"][0, ..., 0].cpu().numpy()
    robot_ids = {int(getattr(o.entity, "per_scene_id", -1))
                 for link in be.agent.robot.links for o in link._objs
                 if getattr(o.entity, "per_scene_id", None) is not None}
    mask = np.isin(seg, list(robot_ids)).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    cv2.imwrite(str(out_path), mask)
    env.close()
    _BAKED.clear()
    print(f"Baked gripper mask {width}x{height} ({(mask > 0).mean() * 100:.1f}% of frame, "
          f"dilate={dilate_px}px) -> {out_path}")
    return mask > 0


# ── Top-level: keep gripper ∪ goal cube, paint the rest ───────────────────────
_RENDERER = None


def get_renderer():
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = GripperRenderer()
    return _RENDERER


# ── Gripper mask: dark-blob detector (real camera, calibration-free) ──────────
# The baked SAPIEN mask assumes the sim wrist-mount calibration; on the real rig
# the camera mount differs, so the baked mask is misaligned. The SO-101 gripper is
# near-black and always enters from the BOTTOM of the wrist frame, so we can detect
# it directly: dark (low V), desaturated, in the lower band, largest blob(s).
GRIP_V_MAX = 70            # gripper pixels are dark: V ≤ this (0-255)
GRIP_S_MAX = 120           # ...and not strongly coloured (excludes a dark-but-saturated cube)
GRIP_LOWER_FRAC = 0.45     # only look below this fraction of the frame height
GRIP_MIN_AREA_FRAC = 5e-4  # ignore dark specks smaller than this fraction of the frame
GRIP_CLOSE_PX = 7          # morphological close to merge the gripper into one blob
GRIP_DILATE_PX = 4         # grow to fully cover the gripper border


def dark_gripper_mask(rgb, hsv=None):
    """Boolean (H,W) mask of the near-black gripper in the lower frame band."""
    h, w = rgb.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    dark = ((V <= GRIP_V_MAX) & (S <= GRIP_S_MAX)).astype(np.uint8) * 255
    dark[:int(GRIP_LOWER_FRAC * h), :] = 0          # gripper is only in the lower band
    if GRIP_CLOSE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GRIP_CLOSE_PX, GRIP_CLOSE_PX))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = np.zeros((h, w), np.uint8)
    min_area = GRIP_MIN_AREA_FRAC * h * w
    for L in range(1, n):
        if stats[L, cv2.CC_STAT_AREA] >= min_area:
            out[lab == L] = 255
    if GRIP_DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GRIP_DILATE_PX + 1, 2 * GRIP_DILATE_PX + 1))
        out = cv2.dilate(out, k)
    return out > 0


def mask_keep_gripper_and_cube(rgb, goal_color, qpos=None,
                               gripper_mode="dark", gripper_dilate_px=2,
                               cube_mode=None, fill="table_mean", return_debug=False):
    """Paint out everything except the gripper and the goal cube.

    rgb: (H,W,3) uint8 RGB. goal_color: int 0-5.
    gripper_mode:
      'dark'  — detect the near-black gripper in the lower band (real camera,
                calibration-free; DEFAULT).
      'baked' — the pose-invariant SAPIEN mask (matches sim, may be misaligned on
                the real rig until the wrist-mount is recalibrated).
      'fk'    — live FK render from qpos (needs qpos; close-range, imperfect).
      'none'  — no gripper mask.
    cube_mode: 'bright_top' (default) or 'full_blob'. fill: 'table_mean' or (r,g,b).
    Returns the masked RGB (and a debug dict if return_debug).
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    if gripper_mode == "dark":
        grip = dark_gripper_mask(rgb, hsv=hsv)
    elif gripper_mode == "baked":
        grip = load_gripper_mask(w, h)
        if grip is None:
            raise ValueError("gripper_mode='baked' but no gripper_mask.png — run --bake.")
    elif gripper_mode == "fk":
        if qpos is None:
            raise ValueError("gripper_mode='fk' needs qpos.")
        grip = get_renderer().render_mask(qpos, w, h, dilate_px=gripper_dilate_px)
    elif gripper_mode == "none":
        grip = np.zeros((h, w), bool)
    else:
        raise ValueError(f"unknown gripper_mode {gripper_mode!r}")

    cube, cube_dbg = goal_cube_mask(rgb, goal_color, hsv=hsv, mode=cube_mode)
    keep = grip | cube

    if fill == "table_mean":
        # Estimate table colour from the lower-centre band (always table on this rig).
        band = rgb[int(0.6 * h):, int(0.3 * w):int(0.7 * w)].reshape(-1, 3)
        fill_rgb = np.median(band, axis=0) if band.size else np.array([180, 180, 180])
    else:
        fill_rgb = np.asarray(fill, dtype=np.float64)

    out = rgb.copy()
    out[~keep] = fill_rgb.astype(rgb.dtype)
    if return_debug:
        return out, {"gripper": grip, "cube": cube, "cube_dbg": cube_dbg}
    return out


# ── Validation against SAPIEN ─────────────────────────────────────────────────
def _validate(seeds=(0, 1, 2, 3), out_dir="debug_artifacts"):
    """Render the gripper silhouette from a live ManiSkill env and overlay it on
    the env's RGB + segmentation, so the user can eyeball the alignment."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401
    import envs.place       # noqa: F401

    W, H = 640, 360
    env = gym.make("SO101PlaceCube-v1", num_envs=1, obs_mode="rgb+segmentation",
                   sensor_configs={"width": W, "height": H},
                   domain_randomization=False)
    rend = get_renderer()
    Path(out_dir).mkdir(exist_ok=True)
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        be = env.unwrapped
        be._update_wrist_camera_pose()
        if be.gpu_sim_enabled:
            be.scene._gpu_apply_all()
            be.scene.px.gpu_update_articulation_kinematics()
            be.scene._gpu_fetch_all()
        # ManiSkill segmentation + rgb for the wrist camera.
        img = be.scene.get_sensor_obs() if hasattr(be.scene, "get_sensor_obs") else None
        sensor = be.scene.sensors["base_camera"]
        cam_obs = sensor.get_obs(rgb=True, segmentation=True, position=False)
        rgb = cam_obs["rgb"][0].cpu().numpy().astype(np.uint8)
        seg = cam_obs["segmentation"][0, ..., 0].cpu().numpy()
        q = be.agent.robot.get_qpos()[0, :6].cpu().numpy()

        my = rend.render_mask(q, W, H, dilate_px=0)
        # robot link ids in seg: anything belonging to the articulation
        robot_ids = set()
        for link in be.agent.robot.links:
            for o in link._objs:
                pid = getattr(o.entity, "per_scene_id", None)
                if pid is not None:
                    robot_ids.add(int(pid))
        sapien_robot = np.isin(seg, list(robot_ids)) if robot_ids else (seg > 0)

        # Per-link colours, so we can see exactly which mesh lands where.
        link_cols = {"gripper_link": (255, 0, 0), "moving_jaw_so101_v1_link": (0, 0, 255),
                     "wrist_link": (255, 255, 0), "lower_arm_link": (0, 255, 255)}
        per_link = rgb.copy()
        for name, parts in rend.link_parts.items():
            single = GripperRenderer.__new__(GripperRenderer)
            single.link_parts = {name: parts}
            mk = single.render_mask(q, W, H, dilate_px=0)
            per_link[mk] = (0.4 * per_link[mk] + 0.6 * np.array(link_cols.get(name, (255, 255, 255)))).astype(np.uint8)

        # Ground-truth dots: project known FK points; they MUST land on the real
        # fingertips/gripper if the camera model is right (isolates mesh issues).
        from so101_fk import fk_frames, finger_positions, tcp_pos
        ext = camera_extrinsic_cv(q); Rcv, tcv = ext[:, :3], ext[:, 3]
        Kk = camera_intrinsic(W, H)
        def proj(p3):
            pc = Rcv @ np.asarray(p3) + tcv
            if pc[2] <= 1e-4:
                return None
            return int(Kk[0, 0] * pc[0] / pc[2] + Kk[0, 2]), int(Kk[1, 1] * pc[1] / pc[2] + Kk[1, 2])
        f1, f2 = finger_positions(q); tcp = tcp_pos(q)
        gorigin = link_transforms(q)["gripper_link"][:3, 3]
        dots = rgb.copy()
        for p3, col in [(f1, (255, 0, 0)), (f2, (0, 0, 255)), (tcp, (0, 255, 0)), (gorigin, (255, 255, 0))]:
            uv = proj(p3)
            if uv is not None:
                cv2.circle(dots, uv, 6, col, -1)

        # Overlay: mine=red, sapien=green, overlap=yellow.
        cmp = rgb.copy()
        cmp[my] = (0.5 * cmp[my] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        cmp[sapien_robot] = (0.5 * cmp[sapien_robot] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        panel = np.concatenate([rgb, dots, per_link, cmp], axis=1)
        inter = (my & sapien_robot).sum()
        union = (my | sapien_robot).sum()
        iou = inter / union if union else 0.0
        path = os.path.join(out_dir, f"gripper_mask_seed{seed}.png")
        cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"seed {seed}: IoU(my, sapien robot)={iou:.3f}  robot_ids={len(robot_ids)} -> {path}")
    env.close()


def _demo(seeds=(0, 1, 2, 3), out_dir="debug_artifacts"):
    """Apply the full keep-mask (baked gripper ∪ goal cube) to sim RGB frames and
    save rgb | keep-mask | masked-result panels, so the user can eyeball the deploy
    behaviour. Uses the baked gripper mask (run --bake first)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import envs.place       # noqa: F401

    W, H = 640, 360
    env = gym.make("SO101PlaceCube-v1", num_envs=1, obs_mode="rgb",
                   sensor_configs={"width": W, "height": H}, domain_randomization=False)
    Path(out_dir).mkdir(exist_ok=True)
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        be = env.unwrapped
        rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)
        goal = int(be.goal_color_idx[0].item()) if hasattr(be, "goal_color_idx") else 0
        masked, dbg = mask_keep_gripper_and_cube(rgb, goal, return_debug=True)
        keep = (dbg["gripper"] | dbg["cube"])
        keep_viz = rgb.copy()
        keep_viz[dbg["gripper"]] = (0.4 * keep_viz[dbg["gripper"]] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        keep_viz[dbg["cube"]] = (0.4 * keep_viz[dbg["cube"]] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        panel = np.concatenate([rgb, keep_viz, masked], axis=1)
        path = os.path.join(out_dir, f"keepmask_seed{seed}.png")
        cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"seed {seed}: goal={goal} kept {keep.mean()*100:.1f}% -> {path}  (red=gripper green=cube)")
    env.close()


COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]


def _live(camera_index=0, goal_color=0, gripper_mode="dark", cube_mode="bright_top",
          width=1920, height=1080, disp_w=640):
    """Live RAW | KEEP-overlay | MASKED viewer on the real wrist camera.

    No robot connection needed. Keys:
      0–5  switch goal colour      g  cycle gripper mode (dark/baked/none)
      b    toggle cube mode (bright_top/full_blob)      q/Esc  quit
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from infer_linux import Cv2Camera

    cam = Cv2Camera(camera_index, width=width, height=height)
    goal = int(goal_color)
    gmodes = ["dark", "baked", "none"]
    gi = gmodes.index(gripper_mode) if gripper_mode in gmodes else 0
    cmode = cube_mode
    print(f"Live mask viewer. goal={goal} ({COLOR_NAMES[goal]}) gripper={gmodes[gi]} cube={cmode}. "
          "Keys: 0-5 goal, g gripper-mode, b cube-mode, q quit.")
    try:
        while True:
            rgb = cam.async_read()                      # (H,W,3) RGB
            masked, dbg = mask_keep_gripper_and_cube(
                rgb, goal, gripper_mode=gmodes[gi], cube_mode=cmode, return_debug=True)
            overlay = rgb.copy()
            overlay[dbg["gripper"]] = (0.4 * overlay[dbg["gripper"]] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            overlay[dbg["cube"]] = (0.4 * overlay[dbg["cube"]] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.putText(overlay, f"goal={goal} {COLOR_NAMES[goal]} | grip={gmodes[gi]} | cube={cmode}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            panel = np.concatenate([rgb, overlay, masked], axis=1)
            dh = int(disp_w * panel.shape[0] / panel.shape[1])
            panel = cv2.resize(panel, (disp_w, dh), interpolation=cv2.INTER_AREA)
            cv2.imshow("RAW | keep (red=gripper green=cube) | MASKED",
                       cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif ord('0') <= k <= ord('5'):
                goal = k - ord('0')
                print(f"goal_color -> {goal} ({COLOR_NAMES[goal]})")
            elif k == ord('g'):
                gi = (gi + 1) % len(gmodes)
                print(f"gripper_mode -> {gmodes[gi]}")
            elif k == ord('b'):
                cmode = "full_blob" if cmode == "bright_top" else "bright_top"
                print(f"cube_mode -> {cmode}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", action="store_true",
                    help="render the pose-invariant gripper mask once from SAPIEN and save it")
    ap.add_argument("--bake_dilate", type=int, default=6,
                    help="dilation (px @ bake res) to pad for wrist-mount calibration error")
    ap.add_argument("--validate", action="store_true",
                    help="compare the live FK silhouette to SAPIEN seg (camera-model check)")
    ap.add_argument("--demo", action="store_true",
                    help="apply the full keep-mask to sim frames and save panels")
    ap.add_argument("--live", action="store_true",
                    help="live RAW | keep | MASKED viewer on the real wrist camera")
    ap.add_argument("--camera", type=int, default=0, help="V4L2 camera index for --live")
    ap.add_argument("--goal_color", type=int, default=0, help="goal cube colour index 0-5 for --live")
    ap.add_argument("--gripper_mode", default="dark", choices=["dark", "baked", "fk", "none"],
                    help="gripper mask source for --live (default: dark = real-camera blob)")
    ap.add_argument("--cube_mode", default="bright_top", choices=["bright_top", "full_blob"],
                    help="cube mask mode for --live")
    args = ap.parse_args()
    if args.bake:
        bake_gripper_mask(dilate_px=args.bake_dilate)
    elif args.validate:
        _validate()
    elif args.demo:
        _demo()
    elif args.live:
        _live(camera_index=args.camera, goal_color=args.goal_color,
              gripper_mode=args.gripper_mode, cube_mode=args.cube_mode)
    else:
        print("Usage: --bake (make the gripper mask) | --live (real-camera viewer) | "
              "--demo (sim keep-mask) | --validate (camera-model check). "
              "Import mask_keep_gripper_and_cube for deploy.")
