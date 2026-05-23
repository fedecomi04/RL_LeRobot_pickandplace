"""Overlay the sim wrist render on a live real wrist frame at the SAME joint pose
to check the calibrated camera extrinsic — the gripper (rigid to the camera)
should coincide if the mount is right.

All hardware + sim are on this machine. The tool:
  1. reads the CURRENT joint angles (torque left ON — arm only holds, never moves)
  2. grabs one real wrist frame
  3. renders the sim wrist at that exact pose, with the NEW (hand-eye) mount and,
     for comparison, the OLD (sim-tuned) mount
  4. saves real | sim-new | overlays (sim edges on real) to debug_artifacts/

    python -m final_utils.overlay_sim_real
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401
import mani_skill.envs  # noqa: F401
import gymnasium as gym
from mani_skill.utils.structs.pose import Pose
from final_utils import bowl_mask as bm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "debug_artifacts")

# OLD sim-tuned mount, for before/after comparison.
OLD_POS = (-0.0006, 0.0498, -0.0641)
OLD_ROT = (np.deg2rad(-90), np.deg2rad(91), np.deg2rad(-35.31))

G_SIM_MIN, G_SIM_MAX = -10.0, 120.0
G_SRV_MIN, G_SRV_MAX = -60.13, 66.73


def _srv_to_sim(s):
    return (s - G_SRV_MIN) / (G_SRV_MAX - G_SRV_MIN) * (G_SIM_MAX - G_SIM_MIN) + G_SIM_MIN


def read_live_qpos_and_frame(camera_index, port, w, h):
    """Read current joints (torque kept ON) + one camera frame."""
    from deploy_utils import infer_linux as il
    from deploy_utils.infer_linux import create_real_robot, Cv2Camera
    from lerobot.motors.motors_bus import MotorNormMode
    il.ROBOT_PORT = port
    robot = create_real_robot()
    robot.connect()                       # does not move the arm; torque unchanged
    bus = robot.bus
    bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    d = bus.sync_read("Present_Position")
    keys = list(d.keys())
    q = np.deg2rad([_srv_to_sim(d[k]) if k == "gripper" else d[k] for k in keys]).astype(np.float64)
    robot.disconnect()

    cam = Cv2Camera(index=camera_index, width=w, height=h, fps=30)
    time.sleep(0.4)
    rgb = np.asarray(cam.async_read())
    cam.close()
    return q, rgb


def _quat_wxyz_from_R(R):
    # SAPIEN/Hamilton wxyz from a rotation matrix
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def render_wrist(u, q, mount_pos_quat=None):
    """Render the sim wrist RGB at joint pose q. If mount_pos_quat=(pos,quat wxyz)
    is given, override the camera mount (else use the env's baked mount)."""
    u.agent.robot.set_qpos(torch.tensor(q, dtype=torch.float32, device=u.device).unsqueeze(0))
    gpu = getattr(u, "gpu_sim_enabled", False)
    if gpu:
        u.scene._gpu_apply_all(); u.scene._gpu_fetch_all()
    if mount_pos_quat is None:
        u._update_wrist_camera_pose()
    else:
        pos, quat = mount_pos_quat
        gp = u.agent.robot.links_map["gripper_link"].pose
        local = Pose.create_from_pq(
            p=torch.tensor(pos, dtype=torch.float32, device=u.device).unsqueeze(0),
            q=torch.tensor(quat, dtype=torch.float32, device=u.device).unsqueeze(0))
        u.wrist_camera_mount.set_pose(gp * local)
    if gpu:
        u.scene._gpu_apply_all(); u.scene._gpu_fetch_all()
    obs = u.get_obs()
    return obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy().astype(np.uint8)


def edge_overlay(real_bgr, sim_rgb, color):
    """Draw the sim render's edges (Canny) onto a copy of the real frame."""
    sim_bgr = cv2.cvtColor(cv2.resize(sim_rgb, (real_bgr.shape[1], real_bgr.shape[0])), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2GRAY), 60, 160)
    out = real_bgr.copy()
    out[edges > 0] = color
    return out


def _base_mount():
    """Current gripper->camera mount (4x4): the hand-eye json if present, else
    the sim's baked WRIST_CAMERA_BASE_* constants."""
    cal = bm._load_mount_calib()
    if cal is not None:
        return cal
    R = bm._quat_to_R(bm._euler_to_quat_wristcam(*bm.WRIST_CAMERA_BASE_ROT_RAD))
    return bm._pose(np.asarray(bm.WRIST_CAMERA_BASE_POS), R)


def adjust_loop(u, q, real):
    """Live nudge of the mount (gripper frame) with the sim gripper blended on
    the real frame. Keys: s=save mount to camera_extrinsics_handeye.json, q=quit."""
    base = _base_mount()
    ctl = "nudge (gripper frame)"
    win = "sim|real gripper overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 960, 540)
    cv2.namedWindow(ctl, cv2.WINDOW_NORMAL); cv2.resizeWindow(ctl, 480, 320)
    for nm in ("dx_mm+50", "dy_mm+50", "dz_mm+50"):
        cv2.createTrackbar(nm, ctl, 50, 100, lambda v: None)            # -50..+50 mm
    for nm in ("droll_0.1d+100", "dpitch_0.1d+100", "dyaw_0.1d+100"):
        cv2.createTrackbar(nm, ctl, 100, 200, lambda v: None)          # -10..+10 deg
    cv2.createTrackbar("blend_%", ctl, 50, 100, lambda v: None)

    def tb(n): return cv2.getTrackbarPos(n, ctl)
    print("ADJUST: align the sim gripper to the real gripper. s=save  q=quit")
    while True:
        nudge_pos = np.array([(tb("dx_mm+50") - 50) / 1000.0,
                              (tb("dy_mm+50") - 50) / 1000.0,
                              (tb("dz_mm+50") - 50) / 1000.0])
        nudge_rpy = np.deg2rad([(tb("droll_0.1d+100") - 100) / 10.0,
                                (tb("dpitch_0.1d+100") - 100) / 10.0,
                                (tb("dyaw_0.1d+100") - 100) / 10.0])
        corr = bm._pose(nudge_pos, bm._quat_to_R(bm._euler_to_quat_wristcam(*nudge_rpy)))
        cand = base @ corr
        sim = render_wrist(u, q, (cand[:3, 3], _quat_wxyz_from_R(cand[:3, :3])))
        sim_bgr = cv2.cvtColor(cv2.resize(sim, (real.shape[1], real.shape[0])), cv2.COLOR_RGB2BGR)
        a = tb("blend_%") / 100.0
        disp = cv2.addWeighted(real, 1 - a, sim_bgr, a, 0)
        disp = edge_overlay(disp, sim, (0, 255, 0))
        txt = (f"nudge mm=({nudge_pos[0]*1000:+.0f},{nudge_pos[1]*1000:+.0f},{nudge_pos[2]*1000:+.0f}) "
               f"deg=({np.rad2deg(nudge_rpy[0]):+.1f},{np.rad2deg(nudge_rpy[1]):+.1f},{np.rad2deg(nudge_rpy[2]):+.1f})")
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(disp, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            import json
            json.dump({"source": "handeye+manual", "T_gripper_cam_mount": cand.tolist()},
                      open(os.path.join(REPO, "calibration", "camera_extrinsics_handeye.json"), "w"), indent=2)
            print(f"saved adjusted mount: pos(mm)={np.round(cand[:3,3]*1000,1)}  "
                  f"(bowl_mask auto-loads it). nudge was {txt}")
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    ap.add_argument("--proc", type=int, nargs=2, default=[360, 640], metavar=("H", "W"))
    ap.add_argument("--adjust", action="store_true",
                    help="live nudge trackbars to align the sim gripper to the real one; s=save mount")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("reading live joint pose + frame (arm holds, not moved) ...")
    q, rgb = read_live_qpos_and_frame(args.camera_index, args.robot_port, 1920, 1080)
    print("q (deg):", np.round(np.rad2deg(q), 1))
    real = cv2.cvtColor(cv2.resize(rgb, (args.proc[1], args.proc[0]), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2BGR)

    env = gym.make("SO101PlaceCube-v1", obs_mode="rgb", sensor_configs={"width": args.proc[1], "height": args.proc[0]},
                   num_envs=1, sim_backend="physx_cuda", domain_randomization=False, reconfiguration_freq=None)
    env.reset(seed=0)
    u = env.unwrapped

    if args.adjust:
        adjust_loop(u, q, real)
        env.close()
        return

    sim_new = render_wrist(u, q)                                    # baked (hand-eye) mount
    old_R = bm._quat_to_R(bm._euler_to_quat_wristcam(*OLD_ROT))
    sim_old = render_wrist(u, q, (np.asarray(OLD_POS), _quat_wxyz_from_R(old_R)))
    env.close()

    cv2.imwrite(os.path.join(OUT, "ov_real.png"), real)
    cv2.imwrite(os.path.join(OUT, "ov_sim_new.png"), cv2.cvtColor(sim_new, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT, "ov_sim_old.png"), cv2.cvtColor(sim_old, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT, "ov_overlay_new.png"), edge_overlay(real, sim_new, (0, 255, 0)))
    cv2.imwrite(os.path.join(OUT, "ov_overlay_old.png"), edge_overlay(real, sim_old, (0, 0, 255)))
    # side-by-side: old overlay | new overlay
    combo = np.hstack([edge_overlay(real, sim_old, (0, 0, 255)),
                       edge_overlay(real, sim_new, (0, 255, 0))])
    cv2.putText(combo, "OLD mount (red edges)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(combo, "NEW mount (green edges)", (args.proc[1] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT, "ov_compare.png"), combo)
    print(f"saved overlays to {OUT}/ov_*.png  (ov_compare.png = old vs new side by side)")


if __name__ == "__main__":
    main()
