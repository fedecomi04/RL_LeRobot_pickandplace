#!/usr/bin/env python3
"""Write a v2-schema synthetic demo file for pipeline smoke-testing.

PURPOSE: lets us validate demo_loader.py + train_squint.py's 50/50 sampler
WITHOUT waiting for the real v2 demo collection (see
logs/2026-05-20_HANDOFF_rlpd-demos-v2-spec.md). The values are random; the
SHAPES and ATTRS match the real spec exactly.

When real v2 demos arrive, replace this file's output with the real h5 and the
trainer doesn't care.

Usage:
    python scripts/make_synthetic_demos.py [--out PATH] [--num-demos N] [--T T]

The script builds the real wrapped env at construction time so it can read the
canonical state_dim and rgb shape — exactly the same query the loader will do.
That way "synthetic but schema-correct" stays correct as envs/place.py evolves.
"""
import argparse
import os
import sys
import time
import math
import json

import numpy as np
import gymnasium as gym
import h5py
import torch

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401 — registers SO101PlaceCube-v1
import mani_skill.envs  # noqa: F401
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
import utils

from mani_skill.utils.wrappers.flatten import (
    FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def build_training_env(num_envs: int = 1):
    """Mirror train_squint.py's env stack so the synthetic demos have the
    canonical shapes."""
    env = gym.make(
        "SO101PlaceCube-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        render_mode="all",
        sim_backend="gpu",
        sensor_configs=dict(width=640, height=360),
        domain_randomization=True,
        domain_randomization_config={
            "shadows": True,
            "num_directional_lights": 3,
            "camera_lag_substeps_range": (0, 0),
        },
        n_distractors=0,
        use_real_bowl=True,
        sim_freq=100,
        control_freq=10,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env = utils.DownsampleObsWrapper(env, target_size=(80, 144))
    env = utils.ColorJitterWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs=num_envs,
                             ignore_terminations=False, record_metrics=True)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/rlpd_demos_synthetic/demos.h5")
    ap.add_argument("--num-demos", type=int, default=5)
    ap.add_argument("--T", type=int, default=670,
                    help="trajectory length per demo (matches real v2 collector)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Building training-stack env to probe canonical shapes...")
    env = build_training_env(num_envs=1)
    obs, _ = env.reset(seed=0)

    rgb_shape = tuple(env.single_observation_space["rgb"].shape)   # (H, W, 3)
    H, W, C = rgb_shape
    state_dim = int(np.prod(env.single_observation_space["state"].shape))
    action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(env.single_action_space.high, dtype=np.float32)
    arm_delta_max = float(action_high[0])
    grip_delta_max = float(action_high[5])

    print(f"  rgb shape  : {rgb_shape}")
    print(f"  state_dim  : {state_dim}")
    print(f"  action low : {action_low}")
    print(f"  action high: {action_high}")
    print(f"  arm bound  : ±{arm_delta_max}, grip bound: ±{grip_delta_max}")

    env.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"\nWriting {args.num_demos} synthetic demos to {args.out}...")
    with h5py.File(args.out, "w") as f:
        # File-level attrs — schema must match v2 spec §3.4
        f.attrs["format_version"] = "2.0"
        f.attrs["env_id"] = "SO101PlaceCube-v1"
        f.attrs["control_mode"] = "pd_joint_target_delta_pos"
        f.attrs["n_distractors"] = 0
        f.attrs["use_real_bowl"] = True
        f.attrs["domain_randomization"] = True
        f.attrs["apply_jitter"] = True
        f.attrs["rgb_h"] = H
        f.attrs["rgb_w"] = W
        f.attrs["state_dim"] = state_dim
        f.attrs["action_dim"] = 6
        f.attrs["arm_delta_max"] = arm_delta_max
        f.attrs["grip_delta_max"] = grip_delta_max
        f.attrs["num_demos"] = args.num_demos
        f.attrs["num_colors"] = 6
        f.attrs["T"] = args.T
        f.attrs["reward_v_min"] = -20.0
        f.attrs["reward_v_max"] = 20.0
        f.attrs["collector_commit"] = "synthetic"
        f.attrs["collected_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())

        for i in range(args.num_demos):
            g = f.create_group(f"demo_{i:03d}")
            # Random RGB in uint8 — encoder doesn't care about content for
            # shape/plumbing tests.
            g.create_dataset("obs/rgb",
                             data=rng.integers(0, 256, (args.T, H, W, C),
                                               dtype=np.uint8),
                             compression="gzip", compression_opts=4)
            # State drawn from N(0,1) — shape correct, values meaningless.
            g.create_dataset("obs/state",
                             data=rng.standard_normal((args.T, state_dim)).astype(np.float32),
                             compression="gzip", compression_opts=4)
            # Actions sampled uniformly inside the env action bounds.
            a = rng.uniform(action_low, action_high,
                            size=(args.T, 6)).astype(np.float32)
            g.create_dataset("actions", data=a,
                             compression="gzip", compression_opts=4)
            # Rewards in a benign range (matches what the env's dense reward
            # produces: a few per-step + a +10 release bonus).
            r = rng.uniform(0.0, 2.0, size=(args.T,)).astype(np.float32)
            r[-50:] += 3.0   # late-trajectory bump (mimics lift+place phase)
            r[-1] += 10.0    # success bonus
            g.create_dataset("rewards", data=r)
            # Terminals: True only at the last step (matches success at end).
            term = np.zeros((args.T,), dtype=bool)
            term[-1] = True
            g.create_dataset("terminals", data=term)
            g.attrs["color_idx"] = int(i % 6)
            g.attrs["cube_pos"] = rng.uniform(-0.3, 0.3, size=(3,))
            g.attrs["bowl_pos"] = rng.uniform(-0.3, 0.3, size=(3,))
            g.attrs["seed"] = int(i)
            g.attrs["success"] = True
            g.attrs["return_sum"] = float(r.sum())

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.num_demos} synthetic demos ({size_mb:.1f} MB)")
    print("This file is for pipeline smoke-testing only — values are random.\n"
          "When real v2 demos arrive, pass --demo_file <real_path> to "
          "train_squint.py instead.")


if __name__ == "__main__":
    main()
