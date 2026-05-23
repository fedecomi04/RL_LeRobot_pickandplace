"""Canonical, reusable SO101 EVAL 2 — two-cube split-then-pick-and-place.

Same task as EVAL 1 (final_utils/pick_place.py) but the scene now has TWO cubes
that may be touching/overlapping. One call splits them apart, then picks the
goal-colour cube and drops it in the bowl:

    from final_utils import split_pick_place
    ok = split_pick_place(goal_color=0, bowl_xy=(0.25, 0.20))   # True if placed

Pipeline (single episode):
    SPLIT — a dedicated 2-cube "split" policy nudges the cubes apart. There is no
            clean state-based "separated" signal, so the phase is timed off the
            arm: run the split policy until the FK end-effector descends below
            --split_below_z (6 cm above the table), then --split_run_s (4 s) more
            seconds, then stop (--split_max_s hard cap). The split policy uses a
            smaller --split_action_scale (0.3). Masking is BACKGROUND-ONLY (table
            mask, no cube-colour mask) — the split policy must see BOTH cubes.
    → return to the rest pose (initial place).
    PICK & PLACE — the unchanged EVAL 1 pipeline: vision RL policy + the FK-gated
            hardcoded grasp, then IK the cube to the bowl and release. Full mask
            stack (background mask + cube-colour mask greying the non-goal cube).

The low-level infra (camera, robot driver, CNN, FK/IK, the grasp/place state
machine) is imported from infer_linux.py and final_utils/pick_place.py so there
is a single source of truth; this module only adds the split phase + the
orchestration of the two phases in one robot session.

Two checkpoints, both with bundled defaults:
    --checkpoint1  the EVAL 1 pick-and-place policy   (default: pick_place_policy.pt)
    --checkpoint2  the 2-cube split policy            (default: runs/split_2cube_quiet_v2_80x144/ckpt.pt)
"""
import argparse
import collections
import datetime
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy_utils import infer_linux as il
from deploy_utils.infer_linux import (
    create_real_robot, RealRobotAgent, CNNEncoder, Actor,
    derive_arch_from_ckpt, preprocess_image, build_state,
    init_viz, log_step, JOINT_NAMES, REST_QPOS, DELTA_CAP, JOINT_LOWER, JOINT_UPPER, CONTROL_HZ,
)
from final_utils.pick_place import run_pick_place, _load_table_calib
from final_utils.hf_record import EpisodeRecorder
from deploy_utils.so101_fk import tcp_pos

try:
    import rerun as rr
except ImportError:
    rr = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DEFAULT_CHECKPOINT1 = os.path.join(_HERE, "pick_place_policy.pt")   # EVAL 1 pick-and-place
# 2-cube split — final checkpoint (split_2cube_quiet_v2, 80x144 input).
DEFAULT_CHECKPOINT2 = os.path.join(_REPO, "runs/split_2cube_quiet_v2_80x144/ckpt.pt")


class RerunWindow:
    """Rolling buffer of the most recent `seconds` of frames + joint scalars,
    written to a Rerun .rrd at the end so the run's tail can be replayed in the
    rerun.io viewer. Spans BOTH eval2 phases (split + pick-and-place).

    Memory stays bounded: the raw camera frame is downsampled to `raw_width`
    before buffering, and entries older than `seconds` are dropped each add().
    A no-op if rerun isn't installed."""

    def __init__(self, seconds=20.0, raw_width=320):
        self.seconds = float(seconds)
        self.raw_width = int(raw_width)
        self.buf = collections.deque()      # (wall_t, raw_rgb, policy_rgb, qpos, target_qpos, action)

    def add(self, raw_rgb, policy_rgb, qpos, target_qpos, action_raw):
        if rr is None:
            return
        t = time.time()
        raw = np.asarray(raw_rgb)
        if raw.shape[1] > self.raw_width:
            rh = int(round(self.raw_width * raw.shape[0] / raw.shape[1]))
            raw = cv2.resize(raw, (self.raw_width, rh), interpolation=cv2.INTER_AREA)
        self.buf.append((t, np.ascontiguousarray(raw).copy(),
                         np.asarray(policy_rgb).copy(), np.asarray(qpos).copy(),
                         np.asarray(target_qpos).copy(), np.asarray(action_raw).copy()))
        while self.buf and (t - self.buf[0][0]) > self.seconds:
            self.buf.popleft()

    def save(self, path):
        """Replay the buffered window into a fresh recording and save it to `path`."""
        if rr is None:
            print("[rrd] rerun not installed — skipping .rrd save.")
            return False
        if not self.buf:
            print("[rrd] no frames buffered — skipping .rrd save.")
            return False
        rec = rr.RecordingStream("squint_eval2_last_window")
        rec.save(str(path))
        span = self.buf[-1][0] - self.buf[0][0]
        for (t, raw, pol, q, tg, a) in self.buf:
            rec.set_time("wall", timestamp=t)
            rec.log("camera/raw", rr.Image(raw))
            rec.log("camera/policy_input", rr.Image(pol))
            for i, name in enumerate(JOINT_NAMES):
                rec.log(f"joints/qpos_measured/{name}", rr.Scalars([float(q[i])]))
                rec.log(f"joints/qpos_target/{name}", rr.Scalars([float(tg[i])]))
                rec.log(f"action_raw/{name}", rr.Scalars([float(a[i])]))
        rec.flush()
        del rec                                  # drop the stream → flush + close the file sink
        print(f"[rrd] saved last {span:.1f}s ({len(self.buf)} frames) → {path}")
        return True


def _load_policy(checkpoint, device):
    """Load encoder+actor and return (encoder, actor, use_bowl_xyz, dims).

    `dims` = (image_h, image_w, cnn_flatten_dim, rgb_proj_dim). The two EVAL 2
    policies may differ in input size, so we DON'T rely on the il.* globals
    persisting between loads — _activate_dims() re-points them per phase."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = derive_arch_from_ckpt(ckpt)
    n_state = arch["n_state"]
    if n_state not in (18, 21):
        raise RuntimeError(f"Unsupported state size in checkpoint: {n_state} (expected 18 or 21)")
    dims = (arch["image_h"], arch["image_w"], arch["cnn_flatten_dim"], arch["rgb_proj_dim"])
    il.IMAGE_H, il.IMAGE_W, il.CNN_FLATTEN_DIM, il.RGB_PROJ_DIM = dims
    encoder = CNNEncoder(layers=arch["layers"]).to(device).eval()
    actor = Actor(n_state=n_state).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"Loaded policy {os.path.basename(checkpoint)} (step {ckpt.get('global_step', '?')}): "
          f"input={dims[0]}×{dims[1]}, n_state={n_state}")
    return encoder, actor, n_state == 21, dims


def _activate_dims(dims):
    """Point infer_linux's preprocess globals at this policy's input size."""
    il.IMAGE_H, il.IMAGE_W, il.CNN_FLATTEN_DIM, il.RGB_PROJ_DIM = dims


def run_split(
    agent, encoder, actor, use_bowl_xyz, goal_color, bowl_xyz,
    *,
    action_scale=0.2,
    below_z=0.06,
    run_s=8.0,
    max_s=20.0,
    viz_on=False,
    device=None,
    frame_sink=None,
):
    """Run the SPLIT phase on an already-connected `agent`.

    Drives the split policy (background-only masking, smaller action_scale) until
    the FK tip first drops below `below_z` above the calibrated table, then
    `run_s` more seconds, then stops (`max_s` hard cap). Leaves the robot wherever
    the split ended — the caller resets to rest before the pick. No return value."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Background mask only: the split policy must see BOTH cubes, so the
    # cube-colour/distractor mask is OFF for this phase (the table/background
    # mask stays on, governed by il.TABLE_MASK_ENABLED).
    prev_distractor = il.COLOR_DISTRACTOR_MASK
    il.COLOR_DISTRACTOR_MASK = False
    try:
        agent.reset(REST_QPOS)
        target_qpos = agent.get_qpos().cpu().numpy().flatten()
        max_steps = max(1, int(round(max_s * CONTROL_HZ)))
        run_steps = max(1, int(round(run_s * CONTROL_HZ)))
        countdown = None                       # steps left once the tip drops below below_z
        # Descent-stall detector — mirrors eval 1's grasp gate. At far reach the FK
        # z is imprecise, so a plain height trigger is unreliable; also start the
        # countdown when the descent PLATEAUS near the table (FK-error-independent).
        min_above, stall_ctr = float("inf"), 0
        stall_steps = max(1, int(round(il.GRASP_STALL_S * CONTROL_HZ)))
        print(f"\n── split: separating 2 cubes (action_scale={action_scale}, "
              f"run {run_s:.1f}s after tip < {below_z*100:.0f} cm, cap {max_s:.0f}s) ──")
        for step in range(max_steps):
            t0 = time.perf_counter()
            qpos = agent.get_qpos().cpu().numpy().flatten()
            agent.capture_sensor_data()
            rgb = agent.get_sensor_data()["base_camera"]["rgb"]

            obs_rgb = preprocess_image(rgb, goal_color=None).to(device)   # background mask only
            obs_state = build_state(qpos, target_qpos, goal_color,
                                    bowl_xyz=bowl_xyz if use_bowl_xyz else None).to(device)
            with torch.no_grad():
                raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()
            action = np.clip(raw_action * action_scale, -1.0, 1.0)
            target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
            agent.set_target_qpos(torch.from_numpy(target_qpos.copy()))

            tcp_xyz = tcp_pos(qpos)
            tcp_r = float(np.hypot(tcp_xyz[0], tcp_xyz[1]))
            tcp_above = float(tcp_xyz[2]) - il.table_z(tcp_xyz[0], tcp_xyz[1])
            # Track descent (reset the stall counter on a new low), like eval 1's gate.
            if tcp_above < min_above - il.GRASP_STALL_EPS:
                min_above, stall_ctr = tcp_above, 0
            else:
                stall_ctr += 1
            stalled = (il.GRASP_GATE_STALL and min_above <= il.GRASP_ENGAGE_Z
                       and stall_ctr >= stall_steps)
            if countdown is None and (tcp_above <= below_z or stalled):
                countdown = run_steps
                why = "stalled" if (stalled and tcp_above > below_z) else "height"
                print(f"  [split] tip reached ({why}): {tcp_above*100:.1f} cm above table "
                      f"→ run {run_s:.1f}s more")

            raw_np = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
            if viz_on:
                log_step(step=step, raw_rgb=raw_np, policy_rgb=obs_rgb[0].cpu().numpy(),
                         qpos=qpos, target_qpos=target_qpos, action_raw=raw_action)
            if frame_sink is not None:
                frame_sink(raw_np, obs_rgb[0].cpu().numpy(), qpos, target_qpos, raw_action)
            if step % 30 == 0:
                print(f"  [split] step {step:4d}  r={tcp_r*100:4.0f}cm  "
                      f"above_table={tcp_above*100:5.1f}cm")

            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
            if countdown is not None:
                countdown -= 1
                if countdown <= 0:
                    print(f"  [split] done at step {step} → returning to rest")
                    break
        else:
            print(f"  [split] hit the {max_s:.0f}s cap (tip never dropped below "
                  f"{below_z*100:.0f} cm) → returning to rest")
    finally:
        il.COLOR_DISTRACTOR_MASK = prev_distractor


def split_pick_place(
    goal_color,
    bowl_xy,
    *,
    checkpoint1=DEFAULT_CHECKPOINT1,
    checkpoint2=DEFAULT_CHECKPOINT2,
    split=True,
    split_action_scale=0.2,
    split_below_z=0.06,
    split_run_s=4.0,
    split_max_s=20.0,
    action_scale=0.15,
    episode_steps=5000,
    place_z=0.10,
    place_open_wait_s=0.5,
    place_speed=0.36,
    robot_port=None,
    camera_index=None,
    viz=True,
    table_mask=True,
    distractor_mask=True,
    save_window=False,
    save_window_s=20.0,
    rrd_path=None,
    record=False,
    upload=True,
    out_dir="deploy_runs",
    hf_repo=None,
    hf_public=False,
):
    """EVAL 2: split the two cubes apart, then pick `goal_color` and drop it in
    the bowl at `bowl_xy`. Both phases run in one robot session.

    If `save_window`, the last `save_window_s` seconds of camera + joint data are
    saved to a Rerun .rrd (`rrd_path`, or an auto-named file in cwd) at the end,
    openable in the rerun.io viewer.

    Returns True iff the cube was grasped, carried to the bowl, and released.
    """
    bowl_xy = np.asarray(bowl_xy, dtype=np.float64).flatten()
    if robot_port is not None:
        il.ROBOT_PORT = robot_port
    if camera_index is not None:
        il.CAMERA_INDEX = camera_index
    il.TABLE_MASK_ENABLED = table_mask
    il.COLOR_DISTRACTOR_MASK = distractor_mask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_table_calib()
    il.load_hue_calib()                                            # measured cube hues, if calibrated

    # Load both policies up front. _load_policy points il's dims at whichever was
    # loaded last, so _activate_dims() re-points them before each phase.
    split_pol = _load_policy(checkpoint2, device) if split else None
    pick_enc, pick_act, pick_bowl, pick_dims = _load_policy(checkpoint1, device)
    policy_bowl_xyz = [float(bowl_xy[0]), float(bowl_xy[1]), 0.0]

    # Rolling window saved to a Rerun .rrd at the end (spans both phases).
    window = RerunWindow(save_window_s) if save_window else None
    # Full-episode recording (raw|masked mp4 + trajectory) → HF, spans both phases.
    recorder = EpisodeRecorder(
        enabled=record, upload=upload, out_dir=out_dir, eval_tier=2,
        hf_repo=hf_repo, hf_public=hf_public,
        meta={"bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
              "checkpoint_pick": os.path.basename(str(checkpoint1)),
              "checkpoint_split": os.path.basename(str(checkpoint2)) if split else None,
              "split": split, "action_scale": action_scale,
              "split_action_scale": split_action_scale},
    )

    def sink(raw, pol, q, tg, a):                 # fan-out: .rrd window + mp4 recorder
        if window is not None:
            window.add(raw, pol, q, tg, a)
        recorder.add(raw, pol, q, tg, a)

    viz_on = init_viz() if viz else False
    robot = create_real_robot()
    robot.connect()
    agent = RealRobotAgent(robot)
    success = False
    try:
        if split:
            split_enc, split_act, split_bowl, split_dims = split_pol
            _activate_dims(split_dims)
            run_split(agent, split_enc, split_act, split_bowl, goal_color, policy_bowl_xyz,
                      action_scale=split_action_scale, below_z=split_below_z,
                      run_s=split_run_s, max_s=split_max_s, viz_on=viz_on,
                      device=device, frame_sink=sink)

        # Phase B — EVAL 1 pick-and-place, run VERBATIM via run_pick_place(): it
        # sets up the eval-1 mask (FastSAM + calibrated colours) itself and runs
        # the same grasp+place state machine. This is the ONLY pick line in eval 2.
        _activate_dims(pick_dims)
        success = run_pick_place(
            agent, pick_enc, pick_act, pick_bowl, goal_color, bowl_xy,
            action_scale=action_scale, episode_steps=episode_steps,
            place_z=place_z, place_open_wait_s=place_open_wait_s,
            place_speed=place_speed, viz_on=viz_on, device=device, frame_sink=sink,
        )
        return success
    finally:
        if window is not None:
            path = rrd_path or f"eval2_last{int(round(save_window_s))}s_{datetime.datetime.now():%Y%m%d_%H%M%S}.rrd"
            try:
                window.save(path)
            except Exception as e:
                print(f"[rrd] save failed ({e}); continuing shutdown.")
        for c in agent.cameras.values():
            try:
                c.close()
            except Exception:
                pass
        agent.reset(REST_QPOS)
        robot.disconnect()
        recorder.finish(success, goal=goal_color)   # encode + upload after teardown


def main():
    p = argparse.ArgumentParser(
        description="SO101 EVAL 2: split the two cubes apart, then pick the colour cube and drop it in the bowl.")
    p.add_argument("--goal_color", type=int, required=True, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--bowl_xy", type=float, nargs=2, default=[0.25, 0.20], metavar=("X", "Y"),
                   help="bowl/goal centre xy in the robot base frame (m); default 0.25 0.20")
    # ── checkpoints (both bundled in final_utils/, so both have defaults) ────
    p.add_argument("--checkpoint1", type=str, default=DEFAULT_CHECKPOINT1,
                   help="EVAL 1 pick-and-place policy (default: bundled pick_place_policy.pt)")
    p.add_argument("--checkpoint2", type=str, default=DEFAULT_CHECKPOINT2,
                   help="2-cube split policy (default: runs/split_2cube_quiet_v2_80x144/ckpt.pt)")
    # ── split phase ─────────────────────────────────────────────────────────
    p.add_argument("--split", action=argparse.BooleanOptionalAction, default=True,
                   help="run the split phase first (default ON; --no-split = plain EVAL 1 pick-and-place)")
    p.add_argument("--split_action_scale", type=float, default=0.2,
                   help="action-scale for the split policy (default 0.2)")
    p.add_argument("--split_below_z", type=float, default=0.06,
                   help="FK tip height (m) above the table; dropping below this starts the split-end countdown (default 0.06)")
    p.add_argument("--split_run_s", type=float, default=4.0,
                   help="seconds to keep splitting after first dropping below --split_below_z (default 3.0)")
    p.add_argument("--split_max_s", type=float, default=20.0,
                   help="hard cap (s) on the split phase if the tip never descends (default 20)")
    # ── pick-and-place (mirrors final_utils/pick_place.py) ──────────────────
    p.add_argument("--action_scale", type=float, default=0.15, help="action-scale for the pick policy")
    p.add_argument("--episode_steps", type=int, default=5000)
    p.add_argument("--place_z", type=float, default=0.10, help="drop height above the table (m)")
    p.add_argument("--place_open_wait_s", type=float, default=0.5, help="hold over bowl before opening (s)")
    p.add_argument("--place_speed", type=float, default=0.36, help="travel speed to the bowl (m/s)")
    p.add_argument("--robot_port", type=str, default=None)
    p.add_argument("--camera_index", type=int, default=None)
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--table_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="grey the background behind the table (default ON)")
    p.add_argument("--distractor_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="grey the non-goal cube during the pick (default ON)")
    # ── Rerun .rrd save of the run's tail ───────────────────────────────────
    p.add_argument("--save_window", action=argparse.BooleanOptionalAction, default=False,
                   help="save the last --save_window_s seconds of camera + joints to a Rerun .rrd (default ON)")
    p.add_argument("--save_window_s", type=float, default=20.0,
                   help="length (s) of the rolling window saved to the .rrd (default 20)")
    p.add_argument("--rrd_path", type=str, default=None,
                   help="output .rrd path (default: auto-named eval2_last<N>s_<timestamp>.rrd in cwd)")
    # ── Episode recording → Hugging Face (default ON) ────────────────────────
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                   help="record the run (raw|masked-policy-input mp4 + trajectory + metadata) and upload to HF (default ON)")
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True,
                   help="upload the recorded run to HF (token from .hf_token/$HF_TOKEN; best-effort, default ON)")
    p.add_argument("--out_dir", type=str, default="deploy_runs", help="local dir for recorded runs")
    p.add_argument("--hf_repo", type=str, default=None, help="target HF dataset repo (default <username>/squint-deploy-runs)")
    p.add_argument("--hf_public", action="store_true", help="create the HF repo public (default private)")
    args = p.parse_args()

    ok = split_pick_place(
        goal_color=args.goal_color, bowl_xy=tuple(args.bowl_xy),
        checkpoint1=args.checkpoint1, checkpoint2=args.checkpoint2,
        split=args.split, split_action_scale=args.split_action_scale,
        split_below_z=args.split_below_z, split_run_s=args.split_run_s,
        split_max_s=args.split_max_s,
        action_scale=args.action_scale, episode_steps=args.episode_steps,
        place_z=args.place_z, place_open_wait_s=args.place_open_wait_s,
        place_speed=args.place_speed, robot_port=args.robot_port,
        camera_index=args.camera_index, viz=args.viz,
        table_mask=args.table_mask, distractor_mask=args.distractor_mask,
        save_window=args.save_window, save_window_s=args.save_window_s,
        rrd_path=args.rrd_path,
        record=args.record, upload=args.upload, out_dir=args.out_dir,
        hf_repo=args.hf_repo, hf_public=args.hf_public,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
