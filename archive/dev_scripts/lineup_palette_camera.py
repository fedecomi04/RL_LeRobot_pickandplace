#!/usr/bin/env python
"""Line the 6 COLOR_PALETTE cubes up in a row on the table and capture them
from the env's WRIST-CAMERA point of view with the robot at its initial
(rest) pose — i.e. exactly what the policy's camera sees at episode start.

Why a subclass: the stock Place env clusters 1 goal + up to 4 distractors
face-to-face (n_distractors is capped at 4 → max 5 cubes), and the scene is
built in __init__. To show all 6 palette colors we keep n_distractors=4
(legal) and add ONE extra cube actor in _load_scene, then override
_initialize_episode to place all six in a straight line and paint them
COLOR_PALETTE[0..5]. Domain randomization is off, so the rendered colors are
the exact palette (no HSV jitter) and lighting is clean.

Runs the GPU sim, so use a CUDA box. From the repo root:

    /home/team44/.conda/envs/squint/bin/python scripts/lineup_palette_camera.py

Env overrides:
    OUT       output dir                       (default: env_rendering)
    LINE_X    row distance in front, metres    (default: 0.18)
    SPACING   centre-to-centre gap, metres     (default: 0.045)
    SHADOWS   "1"/"0" cast-shadow key light    (default: 0)
Outputs $OUT/lineup_wrist.png (camera POV) and $OUT/lineup_ext.png (external).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio.v2 as imageio
import torch
import sapien
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.pose import Pose

import envs  # noqa: F401  registers SO101PlaceCube-v1
import mani_skill.envs  # noqa: F401
from envs.place import PlaceCube, COLOR_PALETTE, NUM_COLORS

OUT = os.environ.get("OUT", "env_rendering")
LINE_X = float(os.environ.get("LINE_X", "0.18"))
SPACING = float(os.environ.get("SPACING", "0.045"))
SHADOWS = os.environ.get("SHADOWS", "0") == "1"
os.makedirs(OUT, exist_ok=True)


class LineupPlaceCube(PlaceCube):
    """PlaceCube + one extra cube so all 6 palette colors can be shown."""

    def _load_scene(self, options):
        super()._load_scene(options)  # builds goal item + 4 distractors
        # One extra single-env cube actor for the 6th palette color. Built here
        # (during reconfigure, before GPU sim init) so it lives in the scene.
        hs = float(self.item_half_sizes[0])
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[hs] * 3)
        builder.add_box_visual(
            half_size=[hs] * 3,
            material=sapien.render.RenderMaterial(
                base_color=[*COLOR_PALETTE[NUM_COLORS - 1], 1.0],
                roughness=0.5, metallic=0.0, specular=0.5,
            ),
        )
        builder.set_scene_idxs([0])
        builder.initial_pose = sapien.Pose(p=[LINE_X, -0.4, hs])
        self.extra_cube = builder.build(name="extra_cube")

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Robot at its EXACT rest pose (initial position) — no qpos noise.
            self.agent.robot.set_qpos(self.rest_qpos.unsqueeze(0).repeat(b, 1))
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            hs = self.item_half_sizes[env_idx]              # (b,)
            base_xy = self.agent.robot.pose.p[env_idx, :2]  # (b, 2)
            qs = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)  # upright

            # Six cubes in a straight row along world +Y (left<->right in the
            # wrist image), centred on the robot's forward axis at x = LINE_X.
            ordered = [self.item] + list(self.distractors) + [self.extra_cube]
            n = len(ordered)  # 6
            y0 = -SPACING * (n - 1) / 2.0
            for i, actor in enumerate(ordered):
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = base_xy[:, 0] + LINE_X
                xyz[:, 1] = base_xy[:, 1] + y0 + i * SPACING
                xyz[:, 2] = hs
                actor.set_pose(Pose.create_from_pq(xyz, qs))

            # Paint palette colors 0..5 left->right (DR off => exact palette).
            idx = lambda v: torch.full((b,), v, dtype=torch.long)
            self.goal_color_idx[env_idx] = 0
            self._set_actor_palette_color(self.item, env_idx, idx(0))
            for k, d in enumerate(self.distractors):
                self.distractor_color_idxs[env_idx, k] = k + 1
                self._set_actor_palette_color(d, env_idx, idx(k + 1))
            # extra_cube already built with COLOR_PALETTE[5]; no recolor needed.

            # Park the bowl + goal marker out of the camera frame.
            far = torch.tensor([-0.6, 0.6, 0.0]).repeat(b, 1)
            self.bin.set_pose(Pose.create_from_pq(far, qs))
            self.goal_site.set_pose(Pose.create_from_pq(far))


def to_img(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0, 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
    return x


def main():
    env = LineupPlaceCube(
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=False,
        domain_randomization_config={"shadows": SHADOWS},
        n_distractors=4,
        sensor_configs=dict(width=288, height=160),
        human_render_camera_configs=dict(shader_pack="default", width=640, height=640),
        sim_freq=100,
        control_freq=10,
    )
    obs, _ = env.reset(seed=0)
    sd = obs["sensor_data"]
    cam_key = "base_camera" if "base_camera" in sd else list(sd.keys())[0]
    wrist_path = os.path.join(OUT, "lineup_wrist.png")
    ext_path = os.path.join(OUT, "lineup_ext.png")
    imageio.imwrite(wrist_path, to_img(sd[cam_key]["rgb"]))
    imageio.imwrite(ext_path, to_img(env.render()))
    env.close()
    print("camera POV ->", wrist_path)
    print("external   ->", ext_path)


if __name__ == "__main__":
    main()
