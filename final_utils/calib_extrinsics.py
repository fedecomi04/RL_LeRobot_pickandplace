"""Hand-eye calibration of the wrist camera (eye-in-hand).

Recovers the TRUE gripper->camera mount transform on the real robot, which
replaces the sim-tuned WRIST_CAMERA_BASE_POS/ROT in final_utils/bowl_mask.py so
the projected bowl mask lands where the real camera actually looks.

Method: fix a calibration board (checkerboard, e.g. shown on a laptop screen,
or an ArUco grid). Hand-move the arm to several poses where the wrist cam sees
the board. At each captured pose we pair
  T_base_gripper  (FK on the joint angles)              — gripper in base
  T_cam_board     (solvePnP of the board)               — board in camera
cv2.calibrateHandEye solves the single rigid X = gripper->camera consistent
across all poses. No need to measure where the board is.

CHECKERBOARD (default). --cols/--rows are INNER-CORNER counts (vertices), not
squares. Square size in mm.
    python -m final_utils.calib_extrinsics --cols 18 --rows 13 --square_mm 13.73
    python -m final_utils.calib_extrinsics --probe          # is the board detected?

ARUCO grid:
    python -m final_utils.calib_extrinsics --board aruco --markers_x 6 --markers_y 6 \
        --marker_mm 31 --sep_mm 9.5

Keys:  c capture · u undo · g solve+save · q/Esc quit
Capture 8-15 poses with VARIED tilt/rotation, board well spread in the frame.
"""
import argparse
import json
import os
import sys
import time

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_utils import bowl_mask as bm
from deploy_utils.so101_fk import fk_frames

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTR_PATH = os.path.join(REPO, "calibration", "camera_intrinsics.json")
CALIB_OUT = os.path.join(REPO, "calibration", "camera_extrinsics_handeye.json")

DICTS = {name: getattr(aruco, name) for name in dir(aruco) if name.startswith("DICT_")}


def _make4x4(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()
    return T


def _rpy_xyz(R):
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2]); y = np.arctan2(-R[2, 0], sy); z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1]); y = np.arctan2(-R[2, 0], sy); z = 0.0
    return np.rad2deg([x, y, z])


def _solve_report(samples, HE, board_tag, method_name):
    """samples: list of (T_base_gripper 4x4, R_target2cam 3x3, t_target2cam 3x1).
    Runs hand-eye, prints the gap + repeatability, writes camera_extrinsics_handeye.json."""
    R_g2b = [s[0][:3, :3] for s in samples]; t_g2b = [s[0][:3, 3] for s in samples]
    R_t2c = [s[1] for s in samples]; t_t2c = [s[2] for s in samples]
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=HE)
    T_go = _make4x4(R_c2g, t_c2g)
    T_gm = T_go @ np.linalg.inv(_make4x4(bm.MOUNT_TO_OPTICAL, np.zeros(3)))
    board_pos = np.array([(s[0] @ T_go @ _make4x4(s[1], s[2]))[:3, 3] for s in samples])
    std_mm = board_pos.std(axis=0) * 1000
    sim_mount = bm._pose(np.asarray(bm.WRIST_CAMERA_BASE_POS),
                         bm._quat_to_R(bm._euler_to_quat_wristcam(*bm.WRIST_CAMERA_BASE_ROT_RAD)))
    delta = np.linalg.inv(sim_mount) @ T_gm
    d_ang = np.rad2deg(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
    print("\n===== HAND-EYE RESULT =====")
    print(f"samples={len(samples)}  board-pos repeatability std(mm)={np.round(std_mm,1)}"
          f"  (>~10mm => add poses / checker 180-deg flip / joint play)")
    print(f"measured mount pos(m)={np.round(T_gm[:3,3],4)} rpy(deg)={np.round(_rpy_xyz(T_gm[:3,:3]),2)}")
    print(f"sim default    pos(m)={np.round(bm.WRIST_CAMERA_BASE_POS,4)} "
          f"rpy(deg)={np.round(_rpy_xyz(sim_mount[:3,:3]),2)}")
    print(f"SIM->REAL GAP: pos {np.round(delta[:3,3]*1000,1)} mm  rotation {d_ang:.2f} deg")
    json.dump({"source": f"handeye_{board_tag}", "method": method_name, "n_samples": len(samples),
               "board_pos_std_mm": std_mm.tolist(), "T_gripper_cam_mount": T_gm.tolist()},
              open(CALIB_OUT, "w"), indent=2)
    print(f"saved {CALIB_OUT}\n")


def _load_samples(save_dir):
    """Reconstruct samples from per-capture json (q, rvec, tvec) — no hardware
    needed, so the solve can run offline."""
    out = []
    for fn in sorted(f for f in os.listdir(save_dir) if f.startswith("sample_") and f.endswith(".json")):
        d = json.loads(open(os.path.join(save_dir, fn)).read())
        T_bg = fk_frames(np.array(d["q"], float))["gripper_link"]
        R_tc, _ = cv2.Rodrigues(np.array(d["rvec"], float))
        out.append((T_bg, R_tc, np.array(d["tvec"], float).reshape(3, 1)))
    return out


# ── Board abstractions: each .detect(gray) -> (ok, rvec, tvec) and draws ───────
class CheckerBoard:
    def __init__(self, cols, rows, square_m, K, dist):
        self.cols, self.rows, self.K, self.dist = cols, rows, K, dist
        self.objp = np.zeros((cols * rows, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
        self.sub = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    def detect(self, gray, disp):
        try:
            found, corners = cv2.findChessboardCornersSB(gray, (self.cols, self.rows))
        except Exception:
            found, corners = cv2.findChessboardCorners(
                gray, (self.cols, self.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return False, None, None, 0
        corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (7, 7), (-1, -1), self.sub)
        cv2.drawChessboardCorners(disp, (self.cols, self.rows), corners, found)
        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist)
        if ok:
            cv2.drawFrameAxes(disp, self.K, self.dist, rvec, tvec, 0.05)
        return ok, rvec, tvec, len(corners)


def _detector_params():
    p = aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return p


class ArucoGrid:
    def __init__(self, mx, my, marker_m, sep_m, dict_name, K, dist):
        self.n = mx * my
        self.K, self.dist = K, dist
        d = aruco.getPredefinedDictionary(DICTS[dict_name])
        self.board = aruco.GridBoard((mx, my), marker_m, sep_m, d)
        self.det = aruco.ArucoDetector(d, _detector_params())

    def detect(self, gray, disp):
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is not None:                                    # drop false positives
            keep = [i for i, m in enumerate(ids.flatten()) if 0 <= int(m) < self.n]
            corners = [corners[i] for i in keep]
            ids = ids[keep] if keep else None
        if ids is None or len(ids) < 4:
            return False, None, None, 0
        aruco.drawDetectedMarkers(disp, corners, ids)
        objp, imgp = self.board.matchImagePoints(corners, ids)
        if objp is None or len(objp) < 4:
            return False, None, None, len(ids)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.dist)
        if ok:
            cv2.drawFrameAxes(disp, self.K, self.dist, rvec, tvec, 0.05)
        return ok, rvec, tvec, len(ids)


def main():
    ap = argparse.ArgumentParser()
    intr = json.loads(open(INTR_PATH).read())
    W0, H0 = intr["image_size"]
    K0 = np.array(intr["K"], float)
    dist = np.array(intr["dist"], float)

    ap.add_argument("--board", default="checker", choices=["checker", "aruco"])
    ap.add_argument("--cols", type=int, default=18, help="checker INNER corners along x")
    ap.add_argument("--rows", type=int, default=13, help="checker INNER corners along y")
    ap.add_argument("--square_mm", type=float, default=151.0 / 11.0)   # 11 squares = 15.1 cm
    ap.add_argument("--markers_x", type=int, default=6)
    ap.add_argument("--markers_y", type=int, default=6)
    ap.add_argument("--marker_mm", type=float, default=31.0)
    ap.add_argument("--sep_mm", type=float, default=9.5)
    ap.add_argument("--dict", default="DICT_6X6_250", choices=list(DICTS))
    ap.add_argument("--camera_index", type=int, default=None)
    ap.add_argument("--robot_port", type=str, default=None)
    ap.add_argument("--probe", action="store_true", help="just report whether the board is detected")
    ap.add_argument("--image", default=None, help="run --probe on a saved image instead of the camera")
    ap.add_argument("--save_dir", default=os.path.join(REPO, "handeye_captures"),
                    help="where each 'c' capture is written (so the solve can re-run offline)")
    ap.add_argument("--solve", default=None, metavar="DIR",
                    help="offline: re-run hand-eye on a saved capture dir (no camera/robot)")
    ap.add_argument("--method", default="park",
                    choices=["tsai", "park", "horaud", "andreff", "daniilidis"])
    args = ap.parse_args()

    HE = {"tsai": cv2.CALIB_HAND_EYE_TSAI, "park": cv2.CALIB_HAND_EYE_PARK,
          "horaud": cv2.CALIB_HAND_EYE_HORAUD, "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
          "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS}[args.method]

    # Offline solve from saved captures — no hardware needed.
    if args.solve:
        samples = _load_samples(args.solve)
        if len(samples) < 3:
            print(f"need >=3 saved samples in {args.solve}, found {len(samples)}"); return
        _solve_report(samples, HE, args.board, args.method)
        return

    def build_board():
        if args.board == "checker":
            return CheckerBoard(args.cols, args.rows, args.square_mm / 1000.0, K0, dist)
        return ArucoGrid(args.markers_x, args.markers_y, args.marker_mm / 1000.0,
                         args.sep_mm / 1000.0, args.dict, K0, dist)

    board = build_board()
    desc = (f"checker {args.cols}x{args.rows} inner, {args.square_mm:.2f}mm/sq"
            if args.board == "checker"
            else f"aruco {args.markers_x}x{args.markers_y} {args.dict}")

    from deploy_utils import infer_linux as il
    from infer_linux import Cv2Camera

    # ── probe (no robot needed) ──
    if args.probe:
        if args.image:
            bgr = cv2.imread(args.image)
            if bgr is None:
                print(f"could not read {args.image}"); return
        else:
            cam = Cv2Camera(index=args.camera_index if args.camera_index is not None else il.CAMERA_INDEX,
                            width=W0, height=H0, fps=30)
            time.sleep(0.5)
            bgr = cv2.cvtColor(np.asarray(cam.async_read()), cv2.COLOR_RGB2BGR)
            cam.close()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ok, rvec, tvec, n = board.detect(gray, bgr)
        print(f"board={desc}  detected={'YES' if ok else 'NO'}  points={n}"
              + (f"  dist={np.linalg.norm(tvec)*100:.1f}cm" if ok else ""))
        if not ok and args.board == "checker":
            print("checker not found — verify --cols/--rows are INNER corners (vertices), try "
                  "swapping them, fill more of the frame, reduce screen glare, hold still.")
        out = os.path.join(REPO, "debug_artifacts", "calib_probe.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, bgr); print(f"saved {out}")
        return

    from final_utils.mask_live import _connect_robot, _read_qpos
    cam_idx = args.camera_index if args.camera_index is not None else il.CAMERA_INDEX
    port = args.robot_port if args.robot_port is not None else il.ROBOT_PORT
    robot, bus, keys = _connect_robot(port)
    cam = Cv2Camera(index=cam_idx, width=W0, height=H0, fps=30)
    print(f"LIVE hand-eye: /dev/video{cam_idx} {W0}x{H0}  board={desc}. Arm FREE — pose by hand.")

    samples = []
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"captures -> {args.save_dir} (re-solvable offline with --solve {args.save_dir})")
    win = "handeye  (c=capture g=solve u=undo q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    try:
        while True:
            rgb = np.asarray(cam.async_read())
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, rvec, tvec, n = board.detect(gray, disp)

            cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(disp, f"{desc}  pts={n} pose={'OK' if ok else '--'}  samples={len(samples)}  "
                              f"c=capture g=solve u=undo q=quit",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("u") and samples:
                samples.pop(); print(f"undo -> {len(samples)} samples")
            elif key == ord("c"):
                if not ok:
                    print("no board pose — reposition"); continue
                q = _read_qpos(bus, keys)
                R_tc, _ = cv2.Rodrigues(rvec)
                samples.append((fk_frames(q)["gripper_link"].copy(), R_tc.copy(), tvec.copy()))
                # Persist so the solve can be re-run offline (e.g. by me).
                idx = len(samples) - 1
                json.dump({"q": np.asarray(q).tolist(),
                           "rvec": np.asarray(rvec).flatten().tolist(),
                           "tvec": np.asarray(tvec).flatten().tolist()},
                          open(os.path.join(args.save_dir, f"sample_{idx:02d}.json"), "w"))
                cv2.imwrite(os.path.join(args.save_dir, f"sample_{idx:02d}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                print(f"captured #{len(samples)} (board {np.linalg.norm(tvec)*100:.1f} cm) -> {args.save_dir}")
            elif key == ord("g"):
                if len(samples) < 3:
                    print("need >=3 samples"); continue
                _solve_report(samples, HE, args.board, args.method)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close(); cv2.destroyAllWindows()
        try:
            bus.enable_torque([k for k in keys if k != "gripper"]); time.sleep(0.3); robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
