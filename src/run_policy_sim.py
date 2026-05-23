"""Minimal headless policy run in sim — no human-render camera (avoids the
SAPIEN GPU shadow/buffer limit hit by env.render()). Runs the policy on the
wrist-cam obs and reports success + cube displacement. Use to just verify a
checkpoint runs, without the real-robot infer scripts.

Usage:
    python run_policy_sim.py --checkpoint runs/<run>/ckpt.pt --seed 0 --max_steps 150
"""
import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Make repo root importable so `import envs` / `from final_utils ...` etc. work
# regardless of how this file is invoked.
import sys as _s_init
from pathlib import Path as _P_init
_s_init.path.insert(0, str(_P_init(__file__).resolve().parent.parent))

import envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import CNNEncoder, Actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101PlaceCube-v1")
    ap.add_argument("--n_distractors", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--no_dr", action="store_true", default=True)
    args = ap.parse_args()

    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=not args.no_dr,
        control_mode="pd_joint_target_delta_pos",
        sensor_configs=dict(width=640, height=480),
        n_distractors=args.n_distractors,
        use_real_bowl=True,
    )
    env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    obs_space = env.unwrapped.single_observation_space
    n_state = obs_space["state"].shape[0]
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(n_obs=(80, 144, 3), device=device).to(device)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval(); actor.eval()
    print(f"loaded {args.checkpoint} @ step {ckpt.get('global_step')}  "
          f"(n_state={n_state}, n_act={n_act})")

    base = env.unwrapped
    color_names = ["red", "blue", "green", "yellow", "purple", "orange"]
    n_success = 0
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        cube_init = base.item.pose.p.detach().cpu().numpy().flatten()
        goal = int(base.goal_color_idx[0].item()) if hasattr(base, "goal_color_idx") else -1
        gname = color_names[goal] if 0 <= goal < 6 else "?"
        succeeded = False
        steps = 0
        for step in range(args.max_steps):
            steps = step + 1
            rgb_now = obs["rgb"]
            state_now = obs["state"]
            if not torch.is_tensor(rgb_now):
                rgb_now = torch.from_numpy(rgb_now)
            if not torch.is_tensor(state_now):
                state_now = torch.from_numpy(state_now)
            rgb_t = rgb_now.permute(0, 3, 1, 2).float()
            rgb_small = F.interpolate(rgb_t, size=(80, 144), mode="area").permute(0, 2, 3, 1).to(torch.uint8)
            with torch.no_grad():
                feats = encoder(rgb_small.to(device))
                mean = actor.forward(feats, state_now.float().to(device))
                action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)
            obs, rew, term, trunc, info = env.step(action.detach().cpu().numpy().astype(np.float32))
            if "success" in info and float(torch.as_tensor(info["success"]).flatten()[0]) > 0.5:
                succeeded = True
            if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
                break
        cube_final = base.item.pose.p.detach().cpu().numpy().flatten()
        disp = float(np.linalg.norm(cube_final - cube_init))
        n_success += int(succeeded)
        print(f"ep {ep+1}/{args.n_episodes}  goal={gname}  steps={steps}  "
              f"success={succeeded}  cube_disp={disp*100:.1f}cm")
    print(f"== {n_success}/{args.n_episodes} succeeded ==")
    env.close()


if __name__ == "__main__":
    main()
