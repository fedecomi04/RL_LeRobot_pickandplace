"""Chaining sim evaluator for the Eval 1/2/3 pipeline.

Matches the user's stated success criteria:
  Task 1 (n_distractors=0):  pick → cube in bowl
  Task 2 (n_distractors=1):  split → reset arm → pick → goal cube in bowl
  Task 3 (n_distractors=3):  split → reset arm → 3× (pick(color_k) → reset arm)
                             → all 3 queried-color cubes in bowl

The env's `evaluate()` only checks `self.item` for cube-in-bowl. For Task 3 we
roll our own checker (`_cube_in_bowl`) that examines every cube's xyz against
the bin bounds and matches by the **initial** per-cube colour (captured before
we mutate goal_color_idx between picks).

Architecture mirrors `scripts/eval_split_policy.py` for the policy/env wiring;
adds (a) phase chaining, (b) soft-reset arm to REST_QPOS between phases via
joint-delta actions, (c) goal_color rotation for Task 3, (d) place-task success
(default env mode, NOT split_only_reward).

Usage on a Brev VM after `conda activate squint`:

  python scripts/eval_pipeline_sim.py --task 1 \
      --pick_ckpt runs/eval1_place_80x144_savageDR_r3/ckpt.pt \
      --n_episodes 50 --out logs/sim_task1_savageDR_r3.json

  python scripts/eval_pipeline_sim.py --task 2 \
      --pick_ckpt final_utils/pick_place_policy.pt \
      --split_ckpt runs/split_2cube_quiet_v2_80x144/ckpt.pt \
      --n_episodes 50 --out logs/sim_task2_quiet_v2.json

  python scripts/eval_pipeline_sim.py --task 3 \
      --pick_ckpt final_utils/pick_place_policy.pt \
      --split_ckpt runs/split_4cube_cf_v1_80x144/ckpt.pt \
      --n_episodes 50 --out logs/sim_task3_cf_v1.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401  (registers SO101PlaceCube-v1)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from train_squint import CNNEncoder, Actor


SIM_CONTROL_HZ = 10  # matches training control_freq

# SO101 "start" keyframe — must match envs/robot/so101.py:230.
REST_QPOS = np.array(
    [0.0, -80.791, 36.747, 86.901, -82.154, 120.0], dtype=np.float32
) * np.pi / 180.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--pick_ckpt", required=True, help="pick policy checkpoint")
    ap.add_argument("--split_ckpt", default=None,
                    help="split policy checkpoint (required for task 2/3)")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_steps", type=int, default=80,
                    help="how many sim steps to run the split policy")
    ap.add_argument("--split_action_scale", type=float, default=0.3)
    ap.add_argument("--pick_steps", type=int, default=150,
                    help="how many sim steps to run the pick policy per attempt")
    ap.add_argument("--pick_action_scale", type=float, default=0.45)
    ap.add_argument("--reset_steps", type=int, default=30,
                    help="steps to soft-reset arm to home between phases")
    ap.add_argument("--image_height", type=int, default=80,
                    help="encoder input H (must match the ckpts' training image size)")
    ap.add_argument("--image_width", type=int, default=144)
    ap.add_argument("--no_dr", action="store_true", default=True,
                    help="disable domain randomization (default ON for cleaner eval)")
    ap.add_argument("--out", required=True, help="output JSON path")
    return ap.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Policy load
# ────────────────────────────────────────────────────────────────────────────

def _derive_n_state(ckpt) -> int:
    """Read the n_state the ckpt was trained with from its actor weights.

    train_squint.Projection has `state_proj[0]` = nn.Linear(n_state, 256), so
    the ckpt's `proj.state_proj.0.weight` has shape (256, n_state).
    """
    actor_sd = ckpt["actor"]
    w = actor_sd.get("proj.state_proj.0.weight")
    if w is None:
        raise RuntimeError(
            f"ckpt has no proj.state_proj.0.weight key. Keys: {list(actor_sd.keys())[:8]}")
    return int(w.shape[1])


def load_policy(ckpt_path, env, image_h, image_w, device):
    """Returns (encoder, actor, n_state) where n_state is the ckpt's expected state dim."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_state = _derive_n_state(ckpt)
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(n_obs=(image_h, image_w, 3), device=device).to(device)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()
    actor.eval()
    step = ckpt.get("global_step", "?")
    print(f"[policy] loaded {ckpt_path} (step {step}, n_state={n_state})", flush=True)
    return encoder, actor, n_state


def policy_action(encoder, actor, obs, image_h, image_w, device, n_state):
    """Returns (E, n_act) numpy action — pre-action_scale tanh output already mapped."""
    rgb = obs["rgb"]
    state = obs["state"]
    if not torch.is_tensor(rgb):
        rgb = torch.from_numpy(rgb)
    if not torch.is_tensor(state):
        state = torch.from_numpy(state)
    # Slice state to the ckpt's expected dim. n_state=18 ckpts (older splits) lack
    # the trailing 3-dim bowl_xyz_robot_frame field; n_state=21 (pick) includes it.
    if state.shape[-1] > n_state:
        state = state[..., :n_state]
    elif state.shape[-1] < n_state:
        pad = torch.zeros(*state.shape[:-1], n_state - state.shape[-1], dtype=state.dtype)
        state = torch.cat([state, pad], dim=-1)
    rgb_t = rgb.permute(0, 3, 1, 2).float()
    rgb_small = F.interpolate(rgb_t, size=(image_h, image_w), mode="area").permute(0, 2, 3, 1).to(torch.uint8)
    with torch.no_grad():
        feats = encoder(rgb_small.to(device))
        mean = actor.forward(feats, state.float().to(device))
        action = torch.tanh(mean) * actor.action_scale + actor.action_bias
    return action.detach().cpu().numpy().astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Per-cube success checker
# ────────────────────────────────────────────────────────────────────────────

def _cube_in_bowl(base, target_color, initial_goal_color, initial_distractor_colors) -> torch.Tensor:
    """For each env, return True iff the cube whose *initial* colour was
    `target_color` is currently above the bowl interior.

    target_color: (E,) long tensor (per env, the colour we want to check).
    initial_goal_color: (E,) — the env's original goal_color_idx (before any mutation).
    initial_distractor_colors: (E, n_distractors) — initial distractor colors.

    Logic mirrors `envs/place.py:1559 evaluate()` for the inside-bin test but
    applied per-cube (not just self.item).
    """
    # Bin centre at table level (z = bin_thickness + cube_half_size — matches env code).
    bin_pos = base.bin.pose.p.clone()
    bin_pos[:, 2] = base.bin_thickness + base.item_half_sizes

    # Stack all cube positions, shape (E, n_cubes, 3).
    cube_pos_list = [base.item.pose.p]
    for d in base.distractors:
        cube_pos_list.append(d.pose.p)
    cube_pos = torch.stack(cube_pos_list, dim=1)

    # Each cube's INITIAL palette colour (immutable per episode).
    color_cols = [initial_goal_color.unsqueeze(1)]
    if initial_distractor_colors.numel() > 0:
        color_cols.append(initial_distractor_colors)
    cube_colors = torch.cat(color_cols, dim=1)  # (E, n_cubes)

    # Per-env mask: which cube has color == target_color?  Each colour appears
    # at most once per episode (palette indices are unique — see _sample_goal).
    is_match = (cube_colors == target_color.unsqueeze(1))  # (E, n_cubes)
    any_match = is_match.any(dim=1)
    # Pick the matched cube's position via masked weighted sum.
    selected_pos = (cube_pos * is_match.unsqueeze(-1).float()).sum(dim=1)

    offset = selected_pos - bin_pos
    inside_x = torch.abs(offset[:, 0]) < base.bin_half_sizes_x
    inside_y = torch.abs(offset[:, 1]) < base.bin_half_sizes_y
    is_above_table = selected_pos[:, 2] > 0.04
    in_bowl = inside_x & inside_y & is_above_table & any_match
    return in_bowl


# ────────────────────────────────────────────────────────────────────────────
# Soft-reset arm to home (no env.reset; cube positions preserved)
# ────────────────────────────────────────────────────────────────────────────

def soft_reset_arm(env, base, n_steps, device):
    """Step the env with joint-delta actions driving qpos toward REST_QPOS.

    Action range is [-1, 1] under control_mode=pd_joint_target_delta_pos;
    we clip (REST - qpos) which behaves as "max delta per step" until the
    arm is within ~1 rad of REST, then settles linearly.
    """
    target = torch.tensor(REST_QPOS, device=device)
    last_obs = None
    for _ in range(n_steps):
        qpos = base.agent.robot.get_qpos()  # (E, 6) radians (clean qpos)
        delta = target.unsqueeze(0) - qpos
        action = torch.clamp(delta, -1.0, 1.0).cpu().numpy().astype(np.float32)
        last_obs, _, _, _, _ = env.step(action)
    return last_obs


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.task >= 2 and args.split_ckpt is None:
        raise SystemExit(f"--task {args.task} requires --split_ckpt")

    n_distractors = {1: 0, 2: 1, 3: 3}[args.task]
    n_picks = 3 if args.task == 3 else 1
    # max_episode_steps must cover the whole chained pipeline + safety margin.
    pipeline_steps = (args.split_steps if args.task >= 2 else 0) \
        + args.reset_steps * n_picks \
        + args.pick_steps * n_picks \
        + 60
    print(f"[setup] task={args.task} n_distractors={n_distractors} n_episodes={args.n_episodes}", flush=True)
    print(f"[setup] step budget per env: {pipeline_steps} (split={args.split_steps if args.task >= 2 else 0}, "
          f"reset×{n_picks}={args.reset_steps * n_picks}, pick×{n_picks}={args.pick_steps * n_picks}, "
          f"margin=60)", flush=True)

    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=not args.no_dr,
        control_mode="pd_joint_target_delta_pos",
        sensor_configs=dict(width=640, height=480),
        n_distractors=n_distractors,
        use_real_bowl=True,
        # Default = place-task success (item_above_bin & static); we override the
        # check ourselves for Task 3, but the env's mode stays default to avoid
        # mis-shaping during the pick phase.
    )
    env = gym.make("SO101PlaceCube-v1", num_envs=args.n_episodes,
                   max_episode_steps=pipeline_steps, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    base = env.unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] SO101PlaceCube-v1 ready (device={device}, "
          f"state_dim={env.unwrapped.single_observation_space['state'].shape[0]}, "
          f"act_dim={env.unwrapped.single_action_space.shape[0]})", flush=True)

    # Reset env once. Per-env auto-reset is disabled because nothing terminates
    # under default success (we never satisfy is_robot_static fully mid-pipeline)
    # and our pipeline_steps > max_episode_steps avoidance keeps the env "live".
    obs, _ = env.reset(seed=args.seed)

    pick_enc, pick_act, pick_n_state = load_policy(
        args.pick_ckpt, env, args.image_height, args.image_width, device
    )
    if args.task >= 2:
        split_enc, split_act, split_n_state = load_policy(
            args.split_ckpt, env, args.image_height, args.image_width, device
        )

    # Capture the IMMUTABLE per-cube colours before any mutation. These let our
    # custom in-bowl checker tell which physical cube currently has colour C
    # even after we've rotated goal_color_idx for the next pick.
    initial_goal_color = base.goal_color_idx.clone()  # (E,)
    initial_distractor_colors = (
        base.distractor_color_idxs.clone() if n_distractors > 0
        else torch.zeros((args.n_episodes, 0), dtype=torch.long, device=device)
    )

    # The N pick targets (palette colour indices, per env). Each one is a colour
    # that EXISTS in the scene this episode (env's goal + first 2 distractors).
    if args.task == 3:
        pick_targets = [initial_goal_color,
                        initial_distractor_colors[:, 0],
                        initial_distractor_colors[:, 1]]
    elif args.task == 2:
        pick_targets = [initial_goal_color]
    else:  # task 1: env's goal_color (single cube, no distractors)
        pick_targets = [initial_goal_color]

    t_wall0 = time.perf_counter()

    # ── PHASE 1: SPLIT (Task 2/3 only) ─────────────────────────────────────
    if args.task >= 2:
        print(f"\n[phase 1/split] {args.split_steps} steps @ scale {args.split_action_scale}", flush=True)
        for t in range(args.split_steps):
            raw = policy_action(split_enc, split_act, obs, args.image_height,
                                args.image_width, device, split_n_state)
            scaled = np.clip(raw * args.split_action_scale, -1.0, 1.0)
            obs, _, _, _, _ = env.step(scaled)
            if t % 20 == 0:
                print(f"  split step {t:3d}", flush=True)

    # ── PHASE 2..2N: PICK (with reset-to-home between each) ────────────────
    per_pick_success = []
    for pick_i, target_color in enumerate(pick_targets):
        # Reset arm to home before EVERY pick attempt (incl. the very first one
        # for Task 1 to standardise start; cheap when arm is already near home).
        print(f"\n[phase reset {pick_i + 1}/{n_picks}] soft-reset arm ({args.reset_steps} steps)", flush=True)
        obs = soft_reset_arm(env, base, args.reset_steps, device)

        # Mutate goal_color_idx so the policy's one-hot input targets this colour.
        # The env's obs["goal_color"] is recomputed from goal_color_idx on each
        # env.step(); we step once with the new value to refresh `obs`.
        base.goal_color_idx[:] = target_color
        zero_act = np.zeros((args.n_episodes, env.unwrapped.single_action_space.shape[0]),
                            dtype=np.float32)
        obs, _, _, _, _ = env.step(zero_act)

        print(f"[phase pick {pick_i + 1}/{n_picks}] target_color[0]={int(target_color[0].item())} "
              f"({args.pick_steps} steps @ scale {args.pick_action_scale})", flush=True)
        for t in range(args.pick_steps):
            raw = policy_action(pick_enc, pick_act, obs, args.image_height,
                                args.image_width, device, pick_n_state)
            scaled = np.clip(raw * args.pick_action_scale, -1.0, 1.0)
            obs, _, _, _, _ = env.step(scaled)
            if t % 30 == 0:
                print(f"  pick step {t:3d}", flush=True)

        in_bowl = _cube_in_bowl(base, target_color, initial_goal_color, initial_distractor_colors)
        per_pick_success.append(in_bowl.cpu().numpy().astype(bool))
        print(f"  [task {args.task}] pick {pick_i + 1}: "
              f"{in_bowl.sum().item()}/{args.n_episodes} cubes placed", flush=True)

    wall = time.perf_counter() - t_wall0

    # ── AGGREGATE ──────────────────────────────────────────────────────────
    per_pick_success = np.stack(per_pick_success, axis=0)  # (n_picks, E)
    task_success = per_pick_success.all(axis=0)            # (E,) — ALL picks landed

    result = {
        "task": args.task,
        "pick_ckpt": args.pick_ckpt,
        "split_ckpt": args.split_ckpt,
        "n_episodes": int(args.n_episodes),
        "n_distractors": n_distractors,
        "n_picks": n_picks,
        "task_success_rate": float(task_success.mean()),
        "per_pick_success_rate": [float(s.mean()) for s in per_pick_success],
        "step_budget": {
            "split": args.split_steps if args.task >= 2 else 0,
            "reset_arm": args.reset_steps,
            "pick": args.pick_steps,
        },
        "image_size": [args.image_height, args.image_width],
        "domain_randomization": not args.no_dr,
        "wall_s": float(wall),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n══ DONE ══", flush=True)
    print(f"  Task {args.task} success: {result['task_success_rate'] * 100:.1f}% "
          f"({task_success.sum()}/{args.n_episodes})", flush=True)
    print(f"  Per-pick rates: {[f'{s * 100:.1f}%' for s in result['per_pick_success_rate']]}", flush=True)
    print(f"  Wall: {wall:.1f} s ({pipeline_steps / wall * args.n_episodes:.0f} env-steps/s)", flush=True)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
