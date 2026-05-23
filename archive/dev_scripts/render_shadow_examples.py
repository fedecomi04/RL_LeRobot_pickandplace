"""Render 10 shadow examples from the SO101PlaceCube-v1 (eval2) scene.

One static scene (2 cubes face-to-face + bowl + arm at rest qpos). Across the
10 examples we vary the lighting via reset seed:

    every example has 3 directional shadow-casters available, but the
    DR-sampled intensity per light varies — fills can dim to ~0, which
    visually collapses 3 → 2 → 1 dominant shadow per scenario. Reset seed
    also varies each light's direction (rng.uniform in base_random_env.py).

Why this layout instead of "1 light env → 2 light env → 3 light env":
    SAPIEN's parallel renderer pre-sizes its shadow-caster pool on first
    scene build. Rebuilding the scene with MORE shadow casters than the
    first one allocated triggers "too many directional lights that cast
    shadows." One env, fixed at 3 casters, sidesteps this cleanly.

For each example we save two PNGs to OUT_DIR:

    {prefix}_shadow_{NN}_topdown.png   — top-down (z=0.7m, looking down)
    {prefix}_shadow_{NN}_wrist.png     — wrist camera (the policy's RGB input)

Run on the Brev VM (shadows=True is fine at num_envs=1).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

N_EXAMPLES = 10


def build_env(num_envs: int, num_dir_lights: int, render_size: int, top_down_pose):
    """Construct an env with shadows ON and the requested directional-light count.

    top_down_pose: sapien.Pose for the human_render_camera (overrides the
    default 3/4 view via human_render_camera_configs dict).
    """
    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=True,
        # Shadows ON; cap directional-light count per env so the parallel
        # renderer's shadow-map allocation stays under SAPIEN's hard ceiling.
        domain_randomization_config={
            "shadows": True,
            "num_directional_lights": num_dir_lights,
            # keep point lights off for clean shadow attribution
            "num_point_lights": 0,
            # widen the intensity ranges a bit so shadows are visible
            "directional_key_intensity_range": (0.80, 1.80),
            "directional_fill_intensity_range": (0.10, 0.50),
            # zero hue jitter for visual consistency across examples
            "light_color_jitter": (1.0, 1.0),
        },
        control_mode="pd_joint_target_delta_pos",
        # Small sensor cameras — SAPIEN's parallel renderer caps the shadow
        # caster pool by n_lights × total_camera_pixels. 3 casters × (640×480
        # sensor + 512×512 render) exceeds the cap; 128×128 + 256×256 fits.
        sensor_configs=dict(width=128, height=128),
        # NB: do NOT pass shader_pack="default" here — that explicit shader
        # pack triggers a heavier shadow allocation and trips the SAPIEN
        # parallel-renderer cap even at 1 env / 3 lights. Letting the env's
        # native shader stay implicit keeps the allocation under budget.
        human_render_camera_configs=dict(
            pose=top_down_pose,
            width=render_size,
            height=render_size,
        ),
        n_distractors=1,
        use_real_bowl=True,
    )
    env = gym.make("SO101PlaceCube-v1", num_envs=num_envs, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    return env


def to_uint8_chw(arr):
    """env.render() output → (E, H, W, 3) uint8 numpy."""
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr.astype(np.uint8)


def label(img_bgr, text):
    H, W = img_bgr.shape[:2]
    cv2.rectangle(img_bgr, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return img_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="directory to write {prefix}_shadow_NN_<view>.png into")
    ap.add_argument("--prefix", required=True,
                    help="filename prefix e.g. 2026-05-20_1700")
    ap.add_argument("--render_size", type=int, default=256)
    ap.add_argument("--seed_base", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-down camera ~70 cm above the spawn cluster, looking straight down.
    # Up-axis hint of (1, 0, 0) puts +x of the world toward the top of the
    # image (i.e. "looking from above, robot base at bottom of frame, cubes
    # toward the top").
    top_down_pose = sapien_utils.look_at(
        eye=[0.25, 0.0, 0.70],
        target=[0.25, 0.0, 0.00],
        up=[1.0, 0.0, 0.0],
    )

    # Build env ONCE — same scene, 3 directional shadow casters, vary lighting
    # per scenario via reset seed.
    env = build_env(num_envs=1, num_dir_lights=3,
                    render_size=args.render_size, top_down_pose=top_down_pose)
    for example_idx in range(N_EXAMPLES):
        print(f"=== example {example_idx:02d} ===", flush=True)
        obs, _ = env.reset(seed=args.seed_base + example_idx)

        topdown = to_uint8_chw(env.render())  # (1, H, W, 3) RGB
        rgb = obs["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        wrist = np.asarray(rgb).astype(np.uint8)  # (1, Hw, Ww, 3)

        tag = f"shadow{example_idx:02d}"
        td_bgr = cv2.cvtColor(topdown[0], cv2.COLOR_RGB2BGR)
        td_bgr = label(td_bgr, f"#{example_idx:02d}  topdown")
        cv2.imwrite(str(out_dir / f"{args.prefix}_{tag}_topdown.png"), td_bgr)

        wr_bgr = cv2.cvtColor(wrist[0], cv2.COLOR_RGB2BGR)
        wr_bgr = cv2.resize(
            wr_bgr,
            (wr_bgr.shape[1] * 2, wr_bgr.shape[0] * 2),
            interpolation=cv2.INTER_NEAREST,
        )
        wr_bgr = label(wr_bgr, f"#{example_idx:02d}  wrist")
        cv2.imwrite(str(out_dir / f"{args.prefix}_{tag}_wrist.png"), wr_bgr)
        print(f"saved example {example_idx:02d}", flush=True)

    env.close()
    print(f"done — {N_EXAMPLES} examples written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
