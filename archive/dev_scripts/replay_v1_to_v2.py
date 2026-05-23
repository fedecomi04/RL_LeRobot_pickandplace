#!/usr/bin/env python3
"""Replay ONE v1 demo through the canonical v2-wrapped env and write a v2-schema h5.

Minimal version per logs/2026-05-20_HANDOFF_rlpd-demos-approachA-replay-v1-via-v2env.md:
no smoke video, no batch, no full-50 loop. Just produce one trajectory in the loader-
compatible format that we can then replay/load.

Usage:
    python scripts/replay_v1_to_v2.py \
        --in logs/2026-05-20_2143_rlpd-pickplace-50demos-6colors.h5 \
        --out /tmp/replay_v1_to_v2_demo0.h5
"""
import argparse
import datetime
import os
import sys

import h5py
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401  - registers SO101PlaceCube-v1
import mani_skill.envs  # noqa: F401
import utils

from mani_skill.utils.wrappers.flatten import (
    FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


ARM_DELTA_MAX = 0.05
GRIP_DELTA_MAX = 0.20


def absolute_to_delta(abs_targets: np.ndarray, q_init: np.ndarray) -> np.ndarray:
    """Convert (T, 6) absolute joint targets to per-step deltas, clipped to controller bounds."""
    deltas = np.empty_like(abs_targets, dtype=np.float32)
    deltas[0] = abs_targets[0].astype(np.float32) - q_init.astype(np.float32)
    deltas[1:] = abs_targets[1:].astype(np.float32) - abs_targets[:-1].astype(np.float32)
    bound = np.array([ARM_DELTA_MAX] * 5 + [GRIP_DELTA_MAX], dtype=np.float32)
    if np.any(np.abs(deltas) > bound + 1e-6):
        n_clipped = int(np.sum(np.abs(deltas) > bound + 1e-6))
        worst_idx = np.unravel_index(np.argmax(np.abs(deltas) - bound), deltas.shape)
        print(f"WARN: {n_clipped} delta values clipped; worst at step {worst_idx[0]} "
              f"joint {worst_idx[1]} value {deltas[worst_idx]:.4f} bound {bound[worst_idx[1]]:.4f}")
    return np.clip(deltas, -bound, bound).astype(np.float32)


def build_env():
    """Build the canonical v2 env stack (DR off for deterministic replay)."""
    e = gym.make(
        "SO101PlaceCube-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="all",
        sim_backend="gpu",
        sensor_configs=dict(width=640, height=360),
        domain_randomization=False,
        n_distractors=0,
        use_real_bowl=True,
        sim_freq=100,
        control_freq=10,
    )
    e = FlattenRGBDObservationWrapper(e, rgb=True, depth=False, state=True)
    e = utils.DownsampleObsWrapper(e, target_size=(80, 144))
    e = utils.ColorJitterWrapper(e)
    if isinstance(e.action_space, gym.spaces.Dict):
        e = FlattenActionSpaceWrapper(e)
    e = ManiSkillVectorEnv(e, num_envs=1, ignore_terminations=False, record_metrics=True)
    return e


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="v1 demos h5 path")
    ap.add_argument("--out", dest="out", required=True, help="output v2 h5 path")
    ap.add_argument("--demo-idx", type=int, default=0, help="which v1 demo to replay")
    args = ap.parse_args()

    print(f"Reading v1 demo_{args.demo_idx:03d} from {args.inp}")
    with h5py.File(args.inp, "r") as f:
        g = f[f"demo_{args.demo_idx:03d}"]
        actions_v1 = g["actions"][:].astype(np.float64)
        color_idx = int(g.attrs["color_idx"])
        seed = int(g.attrs["seed"])
        cube_pos = np.array(g.attrs["cube_pos"], dtype=np.float32)
        bowl_pos_v1 = np.array(g.attrs["bowl_pos"], dtype=np.float32)
    T = int(actions_v1.shape[0])
    print(f"  T={T}, color_idx={color_idx}, seed={seed}")

    env = build_env()

    # Reset with the demo's seed; override the goal color via env reset options
    # (envs/place.py:_sample_goal_and_distractor_colors honors options["goal_color_idx"]).
    obs, _ = env.reset(seed=seed, options={"goal_color_idx": color_idx})

    # 60-step stabilization with zero deltas keeps the controller at robot.qpos.
    zero_act = np.zeros((1, 6), dtype=np.float32)
    for _ in range(60):
        obs, _, _, _, _ = env.step(zero_act)

    # q_init must come from robot.qpos AFTER stabilization (the controller's
    # delta accumulator anchors here, not at the analytic QPOS_START).
    q_init = _to_np(env.unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
    print(f"  q_init: {np.round(q_init, 4)}")

    deltas = absolute_to_delta(actions_v1.astype(np.float32), q_init)

    # Read actual obs shapes from the live obs — the unwrapped space reports
    # the raw render size (360×640), but DownsampleObsWrapper produces 80×144.
    sample_rgb = _to_np(obs["rgb"])[0]
    sample_state = _to_np(obs["state"])[0]
    rgb_h, rgb_w = int(sample_rgb.shape[0]), int(sample_rgb.shape[1])
    n_state = int(sample_state.shape[0])
    print(f"  state_dim={n_state}, rgb={rgb_h}x{rgb_w}")

    rgb_buf = np.empty((T, rgb_h, rgb_w, 3), dtype=np.uint8)
    state_buf = np.empty((T, n_state), dtype=np.float32)
    rew_buf = np.empty((T,), dtype=np.float32)
    term_buf = np.empty((T,), dtype=bool)

    for t in range(T):
        rgb_buf[t] = _to_np(obs["rgb"])[0]
        state_buf[t] = _to_np(obs["state"])[0].astype(np.float32)
        a = deltas[t:t + 1]
        obs, rew, term, trunc, _ = env.step(a)
        rew_buf[t] = float(_to_np(rew)[0])
        done_arr = _to_np(term) | _to_np(trunc)
        term_buf[t] = bool(done_arr[0])

    # Success: cube xy within 5 cm of bowl xy, and cube above bowl z.
    final_cube = _to_np(env.unwrapped.item.pose.p)[0]
    bowl_p = _to_np(env.unwrapped.bin.pose.p)[0]
    dist_xy = float(np.linalg.norm(final_cube[:2] - bowl_p[:2]))
    success = bool(dist_xy < 0.05 and final_cube[2] > bowl_p[2])
    return_sum = float(rew_buf.sum())
    print(f"  final cube xy={np.round(final_cube[:2], 3)}, bowl xy={np.round(bowl_p[:2], 3)}, "
          f"dist={dist_xy:.4f}")
    print(f"  reward range [{float(rew_buf.min()):.3f}, {float(rew_buf.max()):.3f}], "
          f"sum={return_sum:.3f}")
    print(f"  success={success}")

    env.close()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {args.out}")
    with h5py.File(args.out, "w") as f:
        f.attrs["format_version"] = "2.0"
        f.attrs["env_id"] = "SO101PlaceCube-v1"
        f.attrs["control_mode"] = "pd_joint_target_delta_pos"
        f.attrs["n_distractors"] = 0
        f.attrs["use_real_bowl"] = True
        f.attrs["domain_randomization"] = False
        f.attrs["apply_jitter"] = True
        f.attrs["rgb_h"] = int(rgb_h)
        f.attrs["rgb_w"] = int(rgb_w)
        f.attrs["state_dim"] = int(n_state)
        f.attrs["action_dim"] = 6
        f.attrs["arm_delta_max"] = ARM_DELTA_MAX
        f.attrs["grip_delta_max"] = GRIP_DELTA_MAX
        f.attrs["num_demos"] = 1
        f.attrs["num_colors"] = 6
        f.attrs["T"] = int(T)
        f.attrs["reward_v_min"] = -20.0
        f.attrs["reward_v_max"] = 20.0
        f.attrs["collected_at_utc"] = datetime.datetime.utcnow().isoformat()
        f.attrs["source"] = f"replay_v1_to_v2 from {os.path.basename(args.inp)} demo_{args.demo_idx:03d}"

        g = f.create_group("demo_000")
        g.create_dataset("obs/rgb", data=rgb_buf, compression="gzip", compression_opts=4)
        g.create_dataset("obs/state", data=state_buf, compression="gzip", compression_opts=4)
        g.create_dataset("actions", data=deltas, compression="gzip", compression_opts=4)
        g.create_dataset("rewards", data=rew_buf)
        g.create_dataset("terminals", data=term_buf)
        g.attrs["color_idx"] = color_idx
        g.attrs["cube_pos"] = cube_pos
        g.attrs["bowl_pos"] = bowl_pos_v1
        g.attrs["seed"] = seed
        g.attrs["success"] = success
        g.attrs["return_sum"] = return_sum

    print("Done.")


if __name__ == "__main__":
    main()
