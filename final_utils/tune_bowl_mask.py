"""Sim-free validator/tuner for the geometric bowl mask.

Pure numpy + cv2 (+ so101_fk) — NO ManiSkill, NO sim. Projects the bowl cylinder
through the exact same FK + extrinsic chain as final_utils/bowl_mask.py, so what
you see is what deploy masks. Three modes:

  • LIVE (real camera + arm):  validate/tune on incoming frames. Arm goes limp
    so you hand-move it; the cylinder tracks the live wrist pose.
        python -m final_utils.tune_bowl_mask --live --bowl_xy 0.25 -0.06
        python -m final_utils.tune_bowl_mask --live --bowl_xy 0.25 -0.06 --camera_index 1

  • IMAGE (saved frame + its joints):
        python -m final_utils.tune_bowl_mask --image debug_artifacts/wrist_frame.png --q -1 -81 46 75 -94 30

  • SYNTHETIC (default, no hardware): black canvas + projected table grid + axes;
    move the camera with the j0..j5 sliders.

Sliders (window "controls"):
    j0..j5 (synthetic only)                          camera pose
    radius_mm height_mm z_floor_mm  margin_%          cylinder geometry
    dx/dy/dz_mm  droll/dpitch/dyaw                    camera-pose nudge (real-cal)

Keys:  s save nudge+geom to bowl_mask_calib.json · q/Esc quit (prints values)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_utils import bowl_mask as bm

CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration",
                          "bowl_mask_calib.json")

# Default look-down wrist pose (deg, FK order) for synthetic mode.
DEFAULT_Q_DEG = [-1.0, -81.0, 46.0, 75.0, -94.0, 30.0]


def _project(pts_base, q, h, w, nudge):
    return bm.project_points(pts_base, q, h, w, nudge=nudge)


def _draw_polyline_base(img, pts_base, q, nudge, color, thick=1, closed=False):
    h, w = img.shape[:2]
    uv, valid = _project(pts_base, q, h, w, nudge)
    n = len(uv)
    for i in (range(n) if closed else range(n - 1)):
        a, b = i, (i + 1) % n
        if not (valid[a] and valid[b]):
            continue
        pa, pb = uv[a], uv[b]
        if np.all(np.abs(pa) < 1e4) and np.all(np.abs(pb) < 1e4):
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, thick, cv2.LINE_AA)


def draw_table_grid(img, q, nudge, z, x_range=(0.0, 0.55), y_range=(-0.25, 0.25), step=0.05):
    for x in np.arange(x_range[0], x_range[1] + 1e-9, step):
        _draw_polyline_base(img, np.array([[x, y_range[0], z], [x, y_range[1], z]]), q, nudge, (60, 60, 60), 1)
    for y in np.arange(y_range[0], y_range[1] + 1e-9, step):
        _draw_polyline_base(img, np.array([[x_range[0], y, z], [x_range[1], y, z]]), q, nudge, (60, 60, 60), 1)


def draw_axes(img, q, nudge, L=0.05):
    o = np.array([0, 0, 0.0])
    for vec, col in ([L, 0, 0], (0, 0, 255)), ([0, L, 0], (0, 255, 0)), ([0, 0, L], (255, 0, 0)):
        _draw_polyline_base(img, np.array([o, o + np.array(vec)]), q, nudge, col, 2)


def draw_frustum_wire(img, q, nudge, cx, cy, z_floor, base_radius, rim_radius, height, n=48):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    lo = np.column_stack([cx + base_radius * c, cy + base_radius * s, np.full(n, z_floor)])
    hi = np.column_stack([cx + rim_radius * c, cy + rim_radius * s, np.full(n, z_floor + height)])
    _draw_polyline_base(img, lo, q, nudge, (0, 200, 255), 1, closed=True)
    _draw_polyline_base(img, hi, q, nudge, (0, 200, 255), 1, closed=True)
    for i in range(0, n, 6):
        _draw_polyline_base(img, np.array([lo[i], hi[i]]), q, nudge, (0, 140, 200), 1)


def _tb(name, win):
    return cv2.getTrackbarPos(name, win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real camera + arm (incoming frames)")
    ap.add_argument("--camera_index", type=int, default=None)
    ap.add_argument("--robot_port", type=str, default=None)
    ap.add_argument("--cam_width", type=int, default=1920)
    ap.add_argument("--cam_height", type=int, default=1080)
    ap.add_argument("--cam_fps", type=int, default=30)
    ap.add_argument("--image", default=None, help="background wrist frame (png)")
    ap.add_argument("--q", type=float, nargs=6, default=None, metavar="J",
                    help="joint angles (deg, FK order) at image capture")
    ap.add_argument("--bowl_xy", type=float, nargs=2, default=[0.25, 0.0], metavar=("X", "Y"))
    ap.add_argument("--bowl_z", type=float, default=None, help="cylinder floor z (m); default table calib")
    ap.add_argument("--size", type=int, nargs=2, default=[360, 640], metavar=("H", "W"))
    args = ap.parse_args()

    mode = "live" if args.live else ("image" if args.image else "synthetic")
    bowl_xy = np.array(args.bowl_xy, dtype=np.float64)
    # Bowl sits ON the table → base z = 0 by default (NOT table_z_calib, which
    # would launch the cylinder off the table and force a manual reset).
    bowl_z = args.bowl_z if args.bowl_z is not None else 0.0
    q0 = np.array(args.q if args.q is not None else DEFAULT_Q_DEG, dtype=np.float64)

    # ── source setup ──────────────────────────────────────────────────────
    cam = robot = bus = keys = None
    bg = None
    proc_h, proc_w = args.size                       # projection/overlay resolution (16:9)

    if mode == "live":
        from deploy_utils import infer_linux as il
        from deploy_utils.infer_linux import Cv2Camera
        from final_utils.mask_live import _connect_robot, _read_qpos
        cam_idx = args.camera_index if args.camera_index is not None else il.CAMERA_INDEX
        port = args.robot_port if args.robot_port is not None else il.ROBOT_PORT
        robot, bus, keys = _connect_robot(port)
        cam = Cv2Camera(index=cam_idx, width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
        print(f"LIVE: /dev/video{cam_idx}, arm FREE (hand-move it). bowl_xy={tuple(bowl_xy)}")

        def get_frame_q():
            rgb = np.asarray(cam.async_read())
            frame = cv2.cvtColor(cv2.resize(rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_RGB2BGR)
            q = _read_qpos(bus, keys)
            return frame, q
    elif mode == "image":
        bg = cv2.imread(args.image)
        if bg is None:
            sys.exit(f"could not read {args.image}")
        proc_h, proc_w = bg.shape[:2]

        def get_frame_q():
            return bg.copy(), q0
    else:  # synthetic — q from sliders, drawn below

        def get_frame_q():
            q = np.deg2rad([_tb(f"j{i}_deg+180", "controls") - 180 for i in range(6)]).astype(np.float64)
            canvas = np.zeros((proc_h, proc_w, 3), np.uint8)
            draw_table_grid(canvas, q, _read_nudge(), z=bowl_z)
            draw_axes(canvas, q, _read_nudge())
            return canvas, q

    # ── controls ──────────────────────────────────────────────────────────
    ctl = "controls"
    cv2.namedWindow("bowl-mask tune", cv2.WINDOW_NORMAL)
    cv2.namedWindow(ctl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow("bowl-mask tune", min(1280, proc_w * 2), min(720, proc_h * 2))
    cv2.resizeWindow(ctl, 520, 600)

    if mode == "synthetic":
        for i in range(6):
            cv2.createTrackbar(f"j{i}_deg+180", ctl, int(round(q0[i] + 180)), 360, lambda v: None)
    cv2.createTrackbar("rim_mm", ctl, 70, 150, lambda v: None)        # top opening radius
    cv2.createTrackbar("base_mm", ctl, 50, 150, lambda v: None)       # bottom radius
    cv2.createTrackbar("height_mm", ctl, 45, 120, lambda v: None)
    cv2.createTrackbar("z_above_table_mm", ctl, 0, 150, lambda v: None)  # base height above table; 0 = on table
    cv2.createTrackbar("margin_%", ctl, 100, 200, lambda v: None)     # 100 = no inflation
    for nm in ("dx_mm+50", "dy_mm+50", "dz_mm+50"):
        cv2.createTrackbar(nm, ctl, 50, 100, lambda v: None)
    for nm in ("droll_0.1deg+100", "dpitch_0.1deg+100", "dyaw_0.1deg+100"):
        cv2.createTrackbar(nm, ctl, 100, 200, lambda v: None)

    def _read_nudge():
        return np.array([
            (_tb("dx_mm+50", ctl) - 50) / 1000.0,
            (_tb("dy_mm+50", ctl) - 50) / 1000.0,
            (_tb("dz_mm+50", ctl) - 50) / 1000.0,
            np.deg2rad((_tb("droll_0.1deg+100", ctl) - 100) / 10.0),
            np.deg2rad((_tb("dpitch_0.1deg+100", ctl) - 100) / 10.0),
            np.deg2rad((_tb("dyaw_0.1deg+100", ctl) - 100) / 10.0),
        ], dtype=np.float64)

    def read_geom():
        rim = _tb("rim_mm", ctl) / 1000.0
        base = _tb("base_mm", ctl) / 1000.0
        height = _tb("height_mm", ctl) / 1000.0
        z_floor = bowl_z + _tb("z_above_table_mm", ctl) / 1000.0
        margin = max(1, _tb("margin_%", ctl)) / 100.0
        return rim, base, height, z_floor, margin

    print(f"Tuning ({mode}).  s=save  q/Esc=quit")
    try:
        while True:
            disp, q = get_frame_q()
            h, w = disp.shape[:2]
            rim, base, height, z_floor, margin = read_geom()
            nudge = _read_nudge()

            mask, hull = bm.bowl_mask(q, bowl_xy, (h, w), z=z_floor, rim_radius=rim,
                                      base_radius=base, height=height, margin=margin,
                                      nudge=nudge, return_hull=True)
            if mask.any():
                fill = disp.copy()
                fill[mask > 0] = (0.5 * fill[mask > 0] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
                disp = cv2.addWeighted(disp, 0.6, fill, 0.4, 0)
            draw_frustum_wire(disp, q, nudge, bowl_xy[0], bowl_xy[1], z_floor, base, rim, height)
            if hull is not None:
                cv2.polylines(disp, [hull.astype(np.int32)], True, (0, 0, 255), 2)

            txt = (f"rim={rim*1000:.0f} base={base*1000:.0f} h={height*1000:.0f}mm margin={margin:.2f}  "
                   f"nudge=({nudge[0]*1000:+.0f},{nudge[1]*1000:+.0f},{nudge[2]*1000:+.0f})mm "
                   f"({np.rad2deg(nudge[3]):+.1f},{np.rad2deg(nudge[4]):+.1f},{np.rad2deg(nudge[5]):+.1f})d")
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(disp, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("bowl-mask tune", disp)

            key = cv2.waitKey(1 if mode == "live" else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                payload = {"rim_radius": rim, "base_radius": base, "height": height,
                           "z_floor_offset": z_floor - bowl_z, "margin": margin,
                           "nudge": nudge.tolist()}
                json.dump(payload, open(CALIB_PATH, "w"), indent=2)
                print(f"saved {CALIB_PATH}: {payload}")
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                bus.enable_torque([k for k in keys if k != "gripper"])
                import time; time.sleep(0.3)
                robot.disconnect()
            except Exception:
                pass

    rim, base, height, z_floor, margin = read_geom()
    nudge = _read_nudge()
    print("\nFinal:")
    print(f"  rim_radius={rim:.4f} m  base_radius={base:.4f} m  height={height:.4f} m  margin={margin:.2f}")
    print(f"  nudge ={np.round(nudge, 5).tolist()}  (dx,dy,dz m ; droll,dpitch,dyaw rad)")


if __name__ == "__main__":
    main()
