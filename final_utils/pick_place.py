"""Canonical, reusable SO101 pick-and-place.

One call picks the cube of the requested colour and drops it into a bowl:

    from final_utils import pick_and_place
    ok = pick_and_place(goal_color=0, bowl_xy=(0.25, 0.10))   # True if placed

Pipeline (single episode):
    PICK  — vision RL policy + the FK-gated hardcoded grasp (frozen; solved
            2026-05-20). approach → gate (descent-stall / height above the
            reach-calibrated table) → nudge back+down → close → verify by
            gripper stall angle → hold → lift. Retreat+retry on a miss.
    PLACE — IK the (closed) gripper to the bowl centre at a fixed height, wait,
            then open to drop the cube. Success only once the cube is released.

The low-level infra (camera, robot driver, CNN, FK/IK, the grasp constants and
the grasp state machine's building blocks) is imported from infer_linux.py and
so101_fk.py so there is a single source of truth; this module only adds the
orchestration + the place phase.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy_utils import infer_linux as il
from deploy_utils.infer_linux import (
    create_real_robot, RealRobotAgent, CNNEncoder, Actor,
    derive_arch_from_ckpt, preprocess_image, build_state, back_nudge_joint_target,
    init_viz, log_step, REST_QPOS, DELTA_CAP, JOINT_LOWER, JOINT_UPPER, CONTROL_HZ,
)
from deploy_utils.so101_fk import tcp_pos, nudge_arm_joints
from final_utils.hf_record import EpisodeRecorder

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_place_policy.pt")

# ── Tunable grasp-heuristic knobs (wait times + thresholds) ───────────────────
# Each is a (cli_flag, infer_linux global, type, help) row. The DEFAULTS live in
# infer_linux.py (single source of truth); each flag below defaults to None and,
# when you pass it, overrides just that one global for this run — so e.g.
#   python -m final_utils.pick_place --goal_color 1 --bowl_xy .2 -.05 \
#       --grasp_wait_s 0.8 --gate_z 0.006 --close_s 1.3 --hold_s 0.3
# Run with `--help` to see every knob and its meaning.
GRASP_KNOBS = [
    # ── GATE: when to stop the RL approach and start the hardcoded grasp ──
    ("gate_z",          "GRASP_GATE_Z",         float, "base gate height ABOVE the table (m) that triggers the grasp"),
    ("gate_z_slope",    "GRASP_GATE_Z_SLOPE",   float, "extra gate height per metre of reach (m/m); raise so far cubes trigger"),
    ("stall_s",         "GRASP_STALL_S",        float, "s of no further descent before the stall-gate fires"),
    ("stall_eps",       "GRASP_STALL_EPS",      float, "min descent (m) that still counts as 'descending'"),
    ("engage_z",        "GRASP_ENGAGE_Z",       float, "stall-gate is only armed within this height of the table (m)"),
    # ── CLOSE: the corrective nudge + the actual grip ──
    ("grasp_wait_s",    "GRASP_WAIT_S",         float, "s of extra policy run AFTER the gate before closing"),
    ("nudge_m",         "GRASP_NUDGE_M",        float, "corrective back-shift (m) toward base before closing"),
    ("nudge_z",         "GRASP_NUDGE_Z",        float, "target tip height vs table for the nudge (m; negative = press in)"),
    ("nudge_settle_s",  "GRASP_NUDGE_SETTLE_S", float, "s to let the servos reach the nudged pose before closing"),
    ("close_deg",       "GRASP_CLOSE_DEG",      float, "commanded full-close gripper angle (sim deg)"),
    ("close_s",         "GRASP_CLOSE_S",        float, "s allotted for the gripper to close + settle"),
    ("empty_below_deg", "GRASP_EMPTY_BELOW_DEG",float, "grasp-success threshold: measured gripper angle > this = grasped"),
    # ── MISS handling: back off + retry ──
    ("max_retries",     "GRASP_MAX_RETRIES",    int,   "back-off + retry attempts before giving up"),
    ("retreat_up_m",    "GRASP_RETREAT_UP_M",   float, "on a miss, IK the tip UP this far (m)"),
    ("retreat_back_m",  "GRASP_RETREAT_BACK_M", float, "on a miss, IK the tip BACK toward base this far (m)"),
    ("retreat_speed",   "GRASP_RETREAT_SPEED",  float, "tip speed through the back-off move (m/s)"),
    # ── HOLD + LIFT after a confirmed grasp ──
    ("hold_s",          "GRASP_HOLD_S",         float, "s to hold the closed grasp before lifting"),
    ("lift_m",          "GRASP_LIFT_M",         float, "m to raise the cube after a confirmed grasp"),
]


def _apply_grasp_overrides(overrides):
    """Set the infer_linux GRASP_* globals from a {cli_flag: value} dict (skips None)."""
    if not overrides:
        return
    by_flag = {flag: attr for flag, attr, _t, _h in GRASP_KNOBS}
    by_flag["gate_stall"] = "GRASP_GATE_STALL"
    applied = []
    for flag, val in overrides.items():
        if val is None or flag not in by_flag:
            continue
        setattr(il, by_flag[flag], val)
        applied.append(f"{by_flag[flag]}={val}")
    if applied:
        print("Grasp-heuristic overrides: " + ", ".join(applied))


def _load_policy(checkpoint, device):
    """Load encoder+actor and push the architecture into infer_linux's globals
    so preprocess_image / the model classes build to the checkpoint's widths."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = derive_arch_from_ckpt(ckpt)
    n_state = arch["n_state"]
    if n_state not in (18, 21):
        raise RuntimeError(f"Unsupported state size in checkpoint: {n_state} (expected 18 or 21)")
    il.IMAGE_H, il.IMAGE_W = arch["image_h"], arch["image_w"]
    il.CNN_FLATTEN_DIM = arch["cnn_flatten_dim"]
    il.RGB_PROJ_DIM = arch["rgb_proj_dim"]
    encoder = CNNEncoder(layers=arch["layers"]).to(device).eval()
    actor = Actor(n_state=n_state).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"Loaded policy (step {ckpt.get('global_step', '?')}): input={il.IMAGE_H}×{il.IMAGE_W}, "
          f"n_state={n_state}")
    return encoder, actor, n_state == 21


def setup_pick_masking(table_mask=True, distractor_mask=False, fastsam=True,
                       fastsam_weights="FastSAM-s.pt", fastsam_imgsz=320,
                       fastsam_grey=160, fastsam_gripper="sam", fastsam_det_w=320):
    """Configure infer_linux's masking globals for the PICK exactly as eval 1
    (pick_place) does. Single source of truth so eval 2/3 feed the SHARED pick
    policy the identical preprocessed image. The defaults here MUST match
    pick_and_place()'s defaults. FastSAM (default ON) supersedes the HSV table
    mask. NB: this enables FastSAM — call it AFTER the split phase (which needs
    to see both cubes with FastSAM off)."""
    il.TABLE_MASK_ENABLED = table_mask
    il.COLOR_DISTRACTOR_MASK = distractor_mask
    il.FASTSAM_MASK_ENABLED = fastsam
    if fastsam:
        # FastSAM segments + greys; the HSV table/distractor mask is redundant.
        il.TABLE_MASK_ENABLED = False
        il.FASTSAM_WEIGHTS = fastsam_weights
        il.FASTSAM_IMGSZ = fastsam_imgsz
        il.FASTSAM_DET_W = fastsam_det_w       # seg/colour working width (lower = faster)
        il.FASTSAM_GREY = fastsam_grey
        il.FASTSAM_GRIPPER = fastsam_gripper   # "sam" (dark bottom segment), "baked", or "none"


def _load_table_calib():
    """Load the table-plane calibration (radial line or 2D voxel grid) into the
    infer_linux globals. Delegates to il.load_table_z_calib() (single source)."""
    il.load_table_z_calib()


def run_pick_place(
    agent, encoder, actor, use_bowl_xyz, goal_color, bowl_xy,
    *,
    action_scale=0.15,
    episode_steps=5000,
    place_z=0.10,
    place_open_wait_s=0.2,
    place_speed=0.36,
    place_xy_tol=0.015,
    viz_on=False,
    device=None,
    frame_sink=None,
    configure_masking=True,
):
    """Run ONE pick-and-place episode on an already-connected `agent`.

    This is THE eval-1 pick: the FK-gated grasp state machine PLUS the eval-1
    masking (FastSAM keep-mask + calibrated cube colours). Eval 2/3 reuse it
    verbatim — their only pick line is a call to this function with the colour
    and bowl, so the exact same algorithm runs in all three evals.

    configure_masking: when True (default) the eval-1 masking is set up here via
    setup_pick_masking() — so a caller mid-session (eval 2/3, after a split phase
    that ran with a different mask) gets the identical pick-time mask without a
    separate line. pick_and_place() passes False because it already configured
    masking from its own CLI flags.

    The robot setup/teardown and policy load are the caller's responsibility (see
    pick_and_place()); splitting the episode out this way lets eval 2/3 reuse the
    grasp+place state machine + mask instead of forking them.

    `frame_sink`, if given, is called once per step with
    (raw_rgb, policy_rgb, qpos, target_qpos, action_raw) — used by eval2 to keep
    a rolling window for the Rerun .rrd save. None (eval1) = no overhead.

    Returns True iff the cube was grasped, carried to the bowl, and released.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if configure_masking:
        setup_pick_masking()          # eval-1 masking (FastSAM + calibrated colours)
    bowl_xy = np.asarray(bowl_xy, dtype=np.float64).flatten()

    # Place target: bowl centre at a fixed height above the calibrated table.
    place_target = np.array([bowl_xy[0], bowl_xy[1],
                             il.table_z(bowl_xy[0], bowl_xy[1]) + place_z])
    policy_bowl_xyz = [float(bowl_xy[0]), float(bowl_xy[1]), 0.0]   # state z=0 for the policy
    max_place_step = place_speed / CONTROL_HZ                       # Cartesian step per tick (lift)
    TO_BOWL_SPEED_FACTOR = 1.38                                     # carry to the bowl faster than the
                                                                    # lift (1.20 × 1.15 = +15% over the
                                                                    # previous +20% carry speed)
    max_to_bowl_step = place_speed * TO_BOWL_SPEED_FACTOR / CONTROL_HZ

    grasp_close_rad = float(np.deg2rad(il.GRASP_CLOSE_DEG))
    grasp_open_rad = float(np.deg2rad(120.0))                       # full open = drop
    steps = lambda s: max(1, int(round(s * CONTROL_HZ)))

    print(f"\n── pick goal_color={goal_color} → place at bowl xy={bowl_xy} (z={place_z*100:.0f} cm) ──")
    if il.FASTSAM_MASK_ENABLED:
        il.get_fastsam().reset()                 # forget any held cube mask from a prior episode
        il.FASTSAM_PRINT = False                 # fold SAM ms into the per-step perf line below
    agent.reset(REST_QPOS)
    target_qpos = agent.get_qpos().cpu().numpy().flatten()

    # ── Per-step perf logging: control Hz, camera Hz, new frames, SAM ms ──────
    cam = agent.cameras["base_camera"]
    ema_a = 0.2
    ema_ctrl_hz = ema_cam_hz = None
    prev_cam_count = cam.frame_count
    prev_step_t = None
    new_frames = 0

    phase, ctr = "approach", 0
    retries = 0
    lift_z_target = None                                       # tcp z at which the lift phase exits
    min_above, stall_ctr = float("inf"), 0
    result = None                                              # "success" | "failed"

    for step in range(episode_steps):
            t0 = time.perf_counter()

            qpos = agent.get_qpos().cpu().numpy().flatten()
            agent.capture_sensor_data()
            rgb = agent.get_sensor_data()["base_camera"]["rgb"]

            cur_cam_count = cam.frame_count
            new_frames = cur_cam_count - prev_cam_count
            prev_cam_count = cur_cam_count
            if prev_step_t is not None:
                dt = t0 - prev_step_t
                if dt > 0:
                    ic, icam = 1.0 / dt, new_frames / dt
                    ema_ctrl_hz = ic if ema_ctrl_hz is None else (1 - ema_a) * ema_ctrl_hz + ema_a * ic
                    ema_cam_hz = icam if ema_cam_hz is None else (1 - ema_a) * ema_cam_hz + ema_a * icam
            prev_step_t = t0

            il.set_mask_aggressive(qpos)             # aggressive mask when tip is close to the table
            obs_rgb = preprocess_image(rgb, goal_color).to(device)
            obs_state = build_state(qpos, target_qpos, goal_color,
                                    bowl_xyz=policy_bowl_xyz if use_bowl_xyz else None).to(device)
            with torch.no_grad():
                raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()
            action = np.clip(raw_action * action_scale, -1.0, 1.0)

            tcp_xyz = tcp_pos(qpos)
            tcp_r = float(np.hypot(tcp_xyz[0], tcp_xyz[1]))
            z_table = il.table_z(tcp_xyz[0], tcp_xyz[1])
            tcp_above = float(tcp_xyz[2]) - z_table
            gate_z_eff = il.GRASP_GATE_Z + il.GRASP_GATE_Z_SLOPE * tcp_r

            # ── PICK: FK-gated hardcoded grasp (mirrors infer_linux) ──────────
            if phase == "approach":
                target_qpos = np.clip(target_qpos + action * il.DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                if tcp_above < min_above - il.GRASP_STALL_EPS:
                    min_above, stall_ctr = tcp_above, 0
                else:
                    stall_ctr += 1
                stalled = (il.GRASP_GATE_STALL and min_above <= il.GRASP_ENGAGE_Z
                           and stall_ctr >= steps(il.GRASP_STALL_S))
                if tcp_above <= gate_z_eff or stalled:
                    phase, ctr = "wait", 0
                    why = "stalled" if (stalled and tcp_above > gate_z_eff) else "height"
                    print(f"  [grasp] gate ({why}): {tcp_above*100:.1f} cm above table @ r={tcp_r*100:.0f} cm")
            elif phase == "wait":
                target_qpos = np.clip(target_qpos + action * il.DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                ctr += 1
                if ctr >= steps(il.GRASP_WAIT_S):
                    target_qpos, info = back_nudge_joint_target(
                        qpos, target_qpos, il.GRASP_NUDGE_M, z_table + il.GRASP_NUDGE_Z)
                    phase, ctr = "nudge", 0
                    print(f"  [grasp] nudge back {info}")
            elif phase == "nudge":
                ctr += 1
                if ctr >= steps(il.GRASP_NUDGE_SETTLE_S):
                    phase, ctr = "close", 0
                    print("  [grasp] closing")
            elif phase == "close":
                target_qpos[5] = grasp_close_rad
                ctr += 1
                # Wait the full close+settle window, THEN check ONCE: if the jaws
                # stalled above EMPTY_BELOW_DEG a cube is held; if they closed
                # past it the gripper is empty. (Polling every tick false-fired
                # GRASPED before the jaws had closed — even with no cube — so this
                # was reverted to the settle-then-check behaviour.)
                if ctr >= steps(il.GRASP_CLOSE_S):
                    grip_deg = float(np.rad2deg(qpos[5]))
                    if grip_deg > il.GRASP_EMPTY_BELOW_DEG:
                        phase, ctr = "hold", 0
                        print(f"  [grasp] GRASPED ({grip_deg:.1f}°) → hold {il.GRASP_HOLD_S:.1f}s")
                    elif retries < il.GRASP_MAX_RETRIES:
                        retries += 1
                        phase, ctr = "retreat", 0
                        print(f"  [grasp] empty ({grip_deg:.1f}°); returning to initial pose, "
                              f"retry {retries}/{il.GRASP_MAX_RETRIES}")
                    else:
                        result = "failed"
                        print(f"  [grasp] FAILED after {il.GRASP_MAX_RETRIES} retries")
            elif phase == "retreat":
                # On a miss, ramp the arm all the way back to the INITIAL rest pose
                # (gripper open) — not a fixed up/back distance — then rerun the
                # policy from a clean start. Joint-space ramp at ~1 rad/s (rate-
                # independent); stays in the loop so recording keeps capturing.
                step_cap = 1.0 / CONTROL_HZ
                rest = np.asarray(REST_QPOS, dtype=np.float64).flatten()
                delta = rest[:5] - target_qpos[:5]
                if float(np.linalg.norm(delta)) <= 0.02:
                    min_above, stall_ctr = float("inf"), 0
                    phase, ctr = "approach", 0
                    print("  [grasp] back at initial pose → rerunning policy")
                else:
                    target_qpos[:5] = np.clip(target_qpos[:5] + np.clip(delta, -step_cap, step_cap),
                                              JOINT_LOWER[:5], JOINT_UPPER[:5])
                    target_qpos[5] = grasp_open_rad                 # open while returning
            elif phase == "hold":
                target_qpos[5] = grasp_close_rad
                ctr += 1
                if ctr >= steps(il.GRASP_HOLD_S):
                    lift_z_target = float(tcp_xyz[2]) + il.GRASP_LIFT_M
                    phase, ctr = "lift", 0
                    print(f"  [grasp] lifting {il.GRASP_LIFT_M*100:.0f} cm @ {place_speed:.2f} m/s")
            elif phase == "lift":
                # Cartesian lift paced by place_speed (same cap as to_bowl), exit
                # once we've reached the target z — no fixed-time wait.
                dz_remaining = lift_z_target - float(tcp_xyz[2])
                if dz_remaining <= 0.002:
                    phase, ctr = "to_bowl", 0
                    print(f"  [place] carrying to bowl xy={bowl_xy} @ z={place_z*100:.0f} cm")
                else:
                    dz = min(dz_remaining, max_place_step)
                    dq = nudge_arm_joints(qpos, np.array([0.0, 0.0, dz]))
                    target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                    target_qpos[5] = grasp_close_rad
            # ── PLACE: IK to the bowl, wait, then open ────────────────────────
            elif phase == "to_bowl":
                vec = place_target - tcp_xyz
                dist = float(np.linalg.norm(vec))
                if dist <= place_xy_tol:
                    phase, ctr = "drop_wait", 0
                    print(f"  [place] over bowl → wait {place_open_wait_s:.1f}s")
                else:
                    step_vec = vec * min(1.0, max_to_bowl_step / dist)  # carry speed (1.38× lift)
                    dq = nudge_arm_joints(qpos, step_vec)
                    target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                    target_qpos[5] = grasp_close_rad
            elif phase == "drop_wait":
                target_qpos[5] = grasp_close_rad                   # hold over the bowl
                ctr += 1
                if ctr >= steps(place_open_wait_s):
                    phase, ctr = "release", 0
                    print("  [place] opening (drop)")
            elif phase == "release":
                target_qpos[5] = grasp_open_rad                    # open → cube drops
                ctr += 1
                if ctr >= steps(0.7):                              # let it open + fall
                    result = "success"

            agent.set_target_qpos(torch.from_numpy(target_qpos.copy()))

            raw_np = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
            if viz_on:
                log_step(step=step, raw_rgb=raw_np, policy_rgb=obs_rgb[0].cpu().numpy(),
                         qpos=qpos, target_qpos=target_qpos, action_raw=raw_action)
            if frame_sink is not None:
                frame_sink(raw_np, obs_rgb[0].cpu().numpy(), qpos, target_qpos, raw_action)

            if ema_ctrl_hz is not None:
                sam = (f" | SAM {il.FASTSAM_LAST_MS:5.1f}ms {il.FASTSAM_LAST_INFO}"
                       if il.FASTSAM_MASK_ENABLED else "")
                print(f"  step {step:4d} | ctrl {ema_ctrl_hz:5.1f}Hz | cam {ema_cam_hz:4.1f}Hz nf={new_frames}"
                      f"{sam} | r={tcp_r*100:3.0f}cm above={tcp_above*100:5.1f}cm | {phase}")

            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

            if result is not None:
                break

    success = result == "success"
    print(f"\n{'PLACED — success' if success else 'did NOT finish (no place)'} after {step + 1} steps.")
    return success


def pick_and_place(
    goal_color,
    bowl_xy,
    action_scale=0.15,
    episode_steps=5000,
    checkpoint=DEFAULT_CHECKPOINT,
    place_z=0.10,
    place_open_wait_s=0.2,
    place_speed=0.36,
    place_xy_tol=0.015,
    robot_port=None,
    camera_index=None,
    viz=True,
    table_mask=True,
    distractor_mask=False,
    fastsam=True,
    fastsam_weights="FastSAM-s.pt",
    fastsam_imgsz=320,
    fastsam_grey=160,
    fastsam_gripper="sam",
    fastsam_det_w=320,
    grasp_overrides=None,
    arm_vel=None,
    gripper_vel=None,
    record=False,
    upload=True,
    out_dir="deploy_runs",
    hf_repo=None,
    hf_public=False,
):
    """Pick the cube of `goal_color` and drop it into the bowl at `bowl_xy`.

    goal_color: 0 red 1 blue 2 green 3 yellow 4 purple 5 orange.
    bowl_xy:    (x, y) of the bowl centre in the robot base frame (metres).
    place_z:    fixed TCP height above the calibrated table when dropping (m).
    place_open_wait_s: hold over the bowl this long before opening (drop).
    place_speed: Cartesian speed (m/s) the gripper travels to the bowl.

    Sets up the robot/policy/calibration, runs one episode via run_pick_place(),
    then tears down. Returns True iff the cube was grasped, carried, and released.
    """
    if robot_port is not None:
        il.ROBOT_PORT = robot_port
    if camera_index is not None:
        il.CAMERA_INDEX = camera_index
    setup_pick_masking(table_mask=table_mask, distractor_mask=distractor_mask,
                       fastsam=fastsam, fastsam_weights=fastsam_weights,
                       fastsam_imgsz=fastsam_imgsz, fastsam_grey=fastsam_grey,
                       fastsam_gripper=fastsam_gripper, fastsam_det_w=fastsam_det_w)

    _apply_grasp_overrides(grasp_overrides)                        # per-run grasp-heuristic tweaks
    if arm_vel is not None or gripper_vel is not None:             # per-run speed envelope
        il.set_velocity_limits(arm=arm_vel, gripper=gripper_vel)
        print(f"Velocity envelope: arm={il.ARM_VEL_LIMIT} gripper={il.GRIPPER_VEL_LIMIT} rad/s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_table_calib()
    il.load_hue_calib()                                            # measured cube hues, if calibrated
    encoder, actor, use_bowl_xyz = _load_policy(checkpoint, device)
    viz_on = init_viz() if viz else False

    bowl_xy = np.asarray(bowl_xy, dtype=np.float64).flatten()
    recorder = EpisodeRecorder(
        enabled=record, upload=upload, out_dir=out_dir, eval_tier=1,
        hf_repo=hf_repo, hf_public=hf_public,
        meta={"bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
              "checkpoint": os.path.basename(str(checkpoint)),
              "action_scale": action_scale, "place_speed": place_speed,
              "fastsam": fastsam, "fastsam_weights": fastsam_weights if fastsam else None},
    )

    robot = create_real_robot()
    robot.connect()
    agent = RealRobotAgent(robot)
    success = False
    try:
        success = run_pick_place(
            agent, encoder, actor, use_bowl_xyz, goal_color, bowl_xy,
            action_scale=action_scale, episode_steps=episode_steps,
            place_z=place_z, place_open_wait_s=place_open_wait_s,
            place_speed=place_speed, place_xy_tol=place_xy_tol,
            viz_on=viz_on, device=device, frame_sink=recorder.add,
            configure_masking=False,   # already set up from this fn's CLI flags above
        )
        return success
    finally:
        for c in agent.cameras.values():
            try:
                c.close()
            except Exception:
                pass
        agent.reset(REST_QPOS)
        robot.disconnect()
        recorder.finish(success, goal=goal_color)   # encode + upload after teardown (robot idle)


def main():
    p = argparse.ArgumentParser(description="SO101 pick-and-place: pick the colour cube, drop it in the bowl.")
    p.add_argument("--goal_color", type=int, required=True, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--bowl_xy", type=float, nargs=2, required=True, metavar=("X", "Y"),
                   help="bowl centre xy in the robot base frame (m)")
    p.add_argument("--action_scale", type=float, default=0.15,
                   help="scales the policy's approach action (higher = faster/more aggressive approach)")
    p.add_argument("--arm_vel", type=float, default=None,
                   help="arm velocity envelope (rad/s); OVERALL speed knob. Default 3.0 (in infer_linux)")
    p.add_argument("--gripper_vel", type=float, default=None,
                   help="gripper velocity envelope (rad/s); default 9.0")
    p.add_argument("--episode_steps", type=int, default=5000)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--place_z", type=float, default=0.10, help="drop height above the table (m)")
    p.add_argument("--place_open_wait_s", type=float, default=0.2, help="hold over bowl before opening (s)")
    p.add_argument("--place_speed", type=float, default=0.36, help="travel speed to the bowl (m/s)")
    p.add_argument("--place_xy_tol", type=float, default=0.015, help="xy tolerance to count as 'over bowl' (m)")
    p.add_argument("--robot_port", type=str, default=None)
    p.add_argument("--camera_index", type=int, default=None)
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--table_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="grey the background behind the table (default ON)")
    p.add_argument("--distractor_mask", action=argparse.BooleanOptionalAction, default=False,
                   help="grey the non-goal cubes (default OFF; pass --distractor_mask to re-enable)")
    p.add_argument("--fastsam", action=argparse.BooleanOptionalAction, default=True,
                   help="use FastSAM to keep the goal cube + gripper and grey everything else "
                        "(prints the SAM inference time each step; ON by default, overrides "
                        "--table_mask). Pass --no-fastsam to fall back to the HSV mask.")
    p.add_argument("--fastsam_weights", default="FastSAM-s.pt",
                   help="FastSAM-s.pt (fast, ~5ms; default) or FastSAM-x.pt (accurate, ~15ms)")
    p.add_argument("--fastsam_imgsz", type=int, default=320, help="FastSAM input size (lower = faster; default 320)")
    p.add_argument("--fastsam_det_w", type=int, default=320, help="FastSAM seg/colour working width (lower = faster; default 320, was 512)")
    p.add_argument("--fastsam_grey", type=int, default=160, help="constant grey fill (0-255)")
    p.add_argument("--fastsam_gripper", default="sam", choices=["sam", "baked", "none"],
                   help="how to keep the gripper: 'sam' (dark bottom-edge segment from FastSAM, "
                        "any jaw angle; default), 'baked' (gripper_mask.png), or 'none'")

    # ── Grasp heuristic (wait times + thresholds) ─────────────────────────────
    # Each defaults to None = keep the infer_linux.py default; pass one to override
    # it for this run. `--help` prints the current default in parentheses.
    g = p.add_argument_group("grasp heuristic (wait times + thresholds)")
    for flag, attr, typ, helptext in GRASP_KNOBS:
        g.add_argument(f"--{flag}", type=typ, default=None,
                       help=f"{helptext} (default {getattr(il, attr)})")
    g.add_argument("--gate_stall", action=argparse.BooleanOptionalAction, default=None,
                   help=f"also fire the gate when the descent plateaus (default {il.GRASP_GATE_STALL})")

    # ── Episode recording → Hugging Face (default ON) ────────────────────────
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                   help="record the episode (raw|masked-policy-input mp4 + trajectory + metadata) "
                        "to --out_dir and upload to HF (default ON; --no-record to disable)")
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True,
                   help="upload the recorded run to a HF dataset repo (token from .hf_token/$HF_TOKEN; "
                        "best-effort, default ON; --no-upload keeps the run local only)")
    p.add_argument("--out_dir", type=str, default="deploy_runs",
                   help="local dir for recorded runs (default deploy_runs)")
    p.add_argument("--hf_repo", type=str, default=None,
                   help="target HF dataset repo (default <username>/squint-deploy-runs)")
    p.add_argument("--hf_public", action="store_true", help="create the HF repo public (default private)")
    args = p.parse_args()

    grasp_overrides = {flag: getattr(args, flag) for flag, *_ in GRASP_KNOBS}
    grasp_overrides["gate_stall"] = args.gate_stall

    ok = pick_and_place(
        goal_color=args.goal_color, bowl_xy=tuple(args.bowl_xy),
        action_scale=args.action_scale, episode_steps=args.episode_steps,
        checkpoint=args.checkpoint, place_z=args.place_z,
        place_open_wait_s=args.place_open_wait_s, place_speed=args.place_speed,
        place_xy_tol=args.place_xy_tol,
        robot_port=args.robot_port, camera_index=args.camera_index, viz=args.viz,
        table_mask=args.table_mask, distractor_mask=args.distractor_mask,
        fastsam=args.fastsam, fastsam_weights=args.fastsam_weights,
        fastsam_imgsz=args.fastsam_imgsz, fastsam_grey=args.fastsam_grey,
        fastsam_gripper=args.fastsam_gripper, fastsam_det_w=args.fastsam_det_w,
        grasp_overrides=grasp_overrides,
        arm_vel=args.arm_vel, gripper_vel=args.gripper_vel,
        record=args.record, upload=args.upload, out_dir=args.out_dir,
        hf_repo=args.hf_repo, hf_public=args.hf_public,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
