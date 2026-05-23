#!/usr/bin/env python
"""Render a few SO101PlaceCube-v1 frames (wrist/policy + external views) under
domain-randomized lighting, to eyeball the maxed shadows and the camera-shadow
occluder. Runs the GPU sim, so use a CUDA box (not macOS). From the repo root:

    python scripts/render_camera_check.py

Env overrides: OUT (dir), N_FRAMES. Frames -> $OUT/{wrist,ext}_*.png.
  wrist_* = the policy's wrist camera (what the network sees)
  ext_*   = external render_camera (to see the occluder on the gripper)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio.v2 as imageio
import torch
import gymnasium as gym
import envs  # noqa: F401  registers SO101PlaceCube-v1
import mani_skill.envs  # noqa: F401

OUT = os.environ.get("OUT", "/tmp/render_check")
N = int(os.environ.get("N_FRAMES", "6"))
os.makedirs(OUT, exist_ok=True)

env = gym.make(
    "SO101PlaceCube-v1",
    num_envs=1,
    obs_mode="rgb",
    render_mode="rgb_array",
    sim_backend="gpu",
    domain_randomization=True,
    domain_randomization_config={"shadows": True},
    n_distractors=1,
    split_only_reward=True,
    sensor_configs=dict(width=288, height=160),
    human_render_camera_configs=dict(shader_pack="default", width=640, height=640),
    sim_freq=100,
    control_freq=10,
    max_episode_steps=100,
)


def to_img(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0, 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
    return x


for i in range(N):
    obs, _ = env.reset(seed=100 + i)
    sd = obs["sensor_data"]
    cam_key = "base_camera" if "base_camera" in sd else list(sd.keys())[0]
    imageio.imwrite(f"{OUT}/wrist_{i}.png", to_img(sd[cam_key]["rgb"]))
    imageio.imwrite(f"{OUT}/ext_{i}.png", to_img(env.render()))
    print(f"saved frame {i}")

env.close()
print("DONE ->", OUT)
