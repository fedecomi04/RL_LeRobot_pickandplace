"""Canonical SO101 EVAL 3 — four-cube ordered split-then-sequential pick-and-place.

Like EVAL 2 (final_utils/eval2.py) but the scene has FOUR cubes, and the task
queries THREE colours that must be picked and dropped into the bowl IN THAT
EXACT ORDER:

    from final_utils import split_pick_place_sequence
    ok = split_pick_place_sequence(goal_colors=(0, 2, 4), bowl_xy=(0.25, 0.20))

Pipeline (single run):
    SPLIT — a 4-cube split policy nudges the cubes apart, timed off the arm with
            the SAME heuristic as EVAL 2: run until the FK tip drops below
            --split_below_z (6 cm above the table), then --split_run_s more
            seconds, then stop (--split_max_s cap). Background-only masking.
    → return to rest.
    For each colour in `goal_colors`, in order:
        PICK & PLACE — the unchanged EVAL 1 pipeline (vision RL pick policy +
            FK-gated grasp → carry to the bowl → release). Retried until the cube
            is in the bowl (up to --max_attempts_per_color; <=0 = unlimited).
        → return to rest, then move on to the next colour.

Everything is reused from EVAL 1/2 for a single source of truth: the split phase
(`run_split`) and policy/window helpers come from final_utils/eval2.py, and the
grasp+place state machine (`run_pick_place`) from final_utils/pick_place.py. This
module only adds the ordered multi-colour loop.

Two checkpoints, both bundled in final_utils/ with defaults:
    --checkpoint1  the EVAL 1 pick-and-place policy   (default: pick_place_policy.pt)
    --checkpoint2  the 4-cube split policy            (default: split4_policy.pt)
"""
import argparse
import datetime
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy_utils import infer_linux as il
from deploy_utils.infer_linux import create_real_robot, RealRobotAgent, init_viz, REST_QPOS
from final_utils.pick_place import run_pick_place, _load_table_calib
from final_utils.eval2 import _load_policy, _activate_dims, run_split, RerunWindow
from final_utils.hf_record import EpisodeRecorder

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT1 = os.path.join(_HERE, "pick_place_policy.pt")   # EVAL 1 pick-and-place
DEFAULT_CHECKPOINT2 = os.path.join(_HERE, "split4_policy.pt")       # 4-cube split (V7 default)

GOAL_COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]


def split_pick_place_sequence(
    goal_colors,
    bowl_xy,
    *,
    checkpoint1=DEFAULT_CHECKPOINT1,
    checkpoint2=DEFAULT_CHECKPOINT2,
    split=True,
    split_action_scale=0.2,
    split_below_z=0.06,
    split_run_s=8.0,
    split_max_s=20.0,
    action_scale=0.15,
    episode_steps=5000,
    place_z=0.10,
    place_open_wait_s=0.5,
    place_speed=0.36,
    max_attempts_per_color=5,
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
    """EVAL 3: split the four cubes apart, then pick the cubes of `goal_colors`
    (an ordered triple) one at a time, dropping each in the bowl at `bowl_xy`
    before moving to the next. All phases run in one robot session.

    Each colour's pick-and-place is retried until the cube is placed, up to
    `max_attempts_per_color` (<= 0 means retry indefinitely). The robot returns
    to the rest pose between every attempt and every colour.

    Returns True iff ALL requested cubes were placed in order.
    """
    goal_colors = [int(c) for c in goal_colors]
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

    split_pol = _load_policy(checkpoint2, device) if split else None
    pick_enc, pick_act, pick_bowl, pick_dims = _load_policy(checkpoint1, device)
    policy_bowl_xyz = [float(bowl_xy[0]), float(bowl_xy[1]), 0.0]

    # Rolling window saved to a Rerun .rrd at the end (spans the whole run).
    window = RerunWindow(save_window_s) if save_window else None
    # Full-run recording (raw|masked mp4 + trajectory) → HF, spans split + 3 picks.
    recorder = EpisodeRecorder(
        enabled=record, upload=upload, out_dir=out_dir, eval_tier=3,
        hf_repo=hf_repo, hf_public=hf_public,
        meta={"bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
              "checkpoint_pick": os.path.basename(str(checkpoint1)),
              "checkpoint_split": os.path.basename(str(checkpoint2)) if split else None,
              "split": split, "action_scale": action_scale,
              "split_action_scale": split_action_scale,
              "pick_order": [GOAL_COLOR_NAMES[c] for c in goal_colors]},
    )

    def sink(raw, pol, q, tg, a):                 # fan-out: .rrd window + mp4 recorder
        if window is not None:
            window.add(raw, pol, q, tg, a)
        recorder.add(raw, pol, q, tg, a)

    names = " → ".join(GOAL_COLOR_NAMES[c] for c in goal_colors)
    print(f"\n══ EVAL 3: split 4 cubes, then pick in order: {names} → bowl {bowl_xy} ══")

    viz_on = init_viz() if viz else False
    robot = create_real_robot()
    robot.connect()
    agent = RealRobotAgent(robot)
    placed = []
    success = False
    try:
        if split:
            split_enc, split_act, split_bowl, split_dims = split_pol
            _activate_dims(split_dims)
            run_split(agent, split_enc, split_act, split_bowl, goal_colors[0], policy_bowl_xyz,
                      action_scale=split_action_scale, below_z=split_below_z,
                      run_s=split_run_s, max_s=split_max_s, viz_on=viz_on,
                      device=device, frame_sink=sink)

        # Each cube is picked by EVAL 1's run_pick_place(), VERBATIM — it sets up
        # the eval-1 mask (FastSAM + calibrated colours) and runs the same
        # grasp+place state machine. The run_pick_place() call below is the only
        # pick line in eval 3.
        _activate_dims(pick_dims)
        for i, color in enumerate(goal_colors):
            cname = GOAL_COLOR_NAMES[color]
            print(f"\n── EVAL 3 cube {i + 1}/{len(goal_colors)}: {cname} (color {color}) ──")
            ok = False
            attempt = 0
            while not ok:
                attempt += 1
                # Each call resets to the rest pose at its start (= "bring back
                # to initial position" between attempts and between colours).
                ok = run_pick_place(
                    agent, pick_enc, pick_act, pick_bowl, color, bowl_xy,
                    action_scale=action_scale, episode_steps=episode_steps,
                    place_z=place_z, place_open_wait_s=place_open_wait_s,
                    place_speed=place_speed, viz_on=viz_on, device=device, frame_sink=sink,
                )
                if ok:
                    print(f"  [eval3] {cname} placed (attempt {attempt}).")
                elif max_attempts_per_color > 0 and attempt >= max_attempts_per_color:
                    print(f"  [eval3] {cname} NOT placed after {attempt} attempts — giving up.")
                    break
                else:
                    print(f"  [eval3] {cname} not placed (attempt {attempt}); retrying.")
            placed.append(ok)
            if not ok:
                # Order matters; a missed cube can leave the scene wrong for the
                # rest. Stop so the failure is obvious rather than cascading.
                print(f"  [eval3] aborting the sequence after failing on {cname}.")
                break

        all_ok = len(placed) == len(goal_colors) and all(placed)
        success = all_ok
        done = ", ".join(f"{GOAL_COLOR_NAMES[c]}={'OK' if p else 'X'}"
                         for c, p in zip(goal_colors, placed))
        print(f"\n{'EVAL 3 PASS' if all_ok else 'EVAL 3 INCOMPLETE'} — {done}")
        return all_ok
    finally:
        if window is not None:
            path = rrd_path or f"eval3_last{int(round(save_window_s))}s_{datetime.datetime.now():%Y%m%d_%H%M%S}.rrd"
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
        recorder.finish(success, goal=goal_colors)   # encode + upload after teardown


def main():
    p = argparse.ArgumentParser(
        description="SO101 EVAL 3: split four cubes, then pick three colours in order and drop each in the bowl.")
    p.add_argument("--goal_colors", type=int, nargs=3, required=True, metavar=("C1", "C2", "C3"),
                   help="three cube colours to pick IN ORDER (0 red 1 blue 2 green 3 yellow 4 purple 5 orange)")
    p.add_argument("--bowl_xy", type=float, nargs=2, default=[0.25, 0.20], metavar=("X", "Y"),
                   help="bowl/goal centre xy in the robot base frame (m); default 0.25 0.20")
    # ── checkpoints (both bundled in final_utils/, so both have defaults) ────
    p.add_argument("--checkpoint1", type=str, default=DEFAULT_CHECKPOINT1,
                   help="EVAL 1 pick-and-place policy (default: bundled pick_place_policy.pt)")
    p.add_argument("--checkpoint2", type=str, default=DEFAULT_CHECKPOINT2,
                   help="4-cube split policy (default: bundled split4_policy.pt)")
    # ── split phase (same heuristic as EVAL 2) ──────────────────────────────
    p.add_argument("--split", action=argparse.BooleanOptionalAction, default=True,
                   help="run the split phase first (default ON; --no-split = sequential pick only)")
    p.add_argument("--split_action_scale", type=float, default=0.2,
                   help="action-scale for the split policy (default 0.2)")
    p.add_argument("--split_below_z", type=float, default=0.06,
                   help="FK tip height (m) above the table; dropping below this starts the split-end countdown (default 0.06)")
    p.add_argument("--split_run_s", type=float, default=8.0,
                   help="seconds to keep splitting after first dropping below --split_below_z (default 8.0)")
    p.add_argument("--split_max_s", type=float, default=20.0,
                   help="hard cap (s) on the split phase if the tip never descends (default 20)")
    # ── per-colour pick-and-place ───────────────────────────────────────────
    p.add_argument("--max_attempts_per_color", type=int, default=5,
                   help="retry each colour's pick-and-place up to this many times (<=0 = unlimited; default 5)")
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
                   help="grey the non-goal cubes during each pick (default ON; needed to pick the right colour among 4 cubes)")
    # ── Rerun .rrd save of the run's tail ───────────────────────────────────
    p.add_argument("--save_window", action=argparse.BooleanOptionalAction, default=False,
                   help="save the last --save_window_s seconds of camera + joints to a Rerun .rrd (default ON)")
    p.add_argument("--save_window_s", type=float, default=20.0,
                   help="length (s) of the rolling window saved to the .rrd (default 20)")
    p.add_argument("--rrd_path", type=str, default=None,
                   help="output .rrd path (default: auto-named eval3_last<N>s_<timestamp>.rrd in cwd)")
    # ── Episode recording → Hugging Face (default ON) ────────────────────────
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                   help="record the run (raw|masked-policy-input mp4 + trajectory + metadata) and upload to HF (default ON)")
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True,
                   help="upload the recorded run to HF (token from .hf_token/$HF_TOKEN; best-effort, default ON)")
    p.add_argument("--out_dir", type=str, default="deploy_runs", help="local dir for recorded runs")
    p.add_argument("--hf_repo", type=str, default=None, help="target HF dataset repo (default <username>/squint-deploy-runs)")
    p.add_argument("--hf_public", action="store_true", help="create the HF repo public (default private)")
    args = p.parse_args()

    ok = split_pick_place_sequence(
        goal_colors=tuple(args.goal_colors), bowl_xy=tuple(args.bowl_xy),
        checkpoint1=args.checkpoint1, checkpoint2=args.checkpoint2,
        split=args.split, split_action_scale=args.split_action_scale,
        split_below_z=args.split_below_z, split_run_s=args.split_run_s,
        split_max_s=args.split_max_s, max_attempts_per_color=args.max_attempts_per_color,
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
