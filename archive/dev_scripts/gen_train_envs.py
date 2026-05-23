"""Generate numbered env frames produced EXACTLY as in training.

Mirrors train_squint.py's env construction (current defaults) so the lighting /
image-pipeline DR each frame sees is the same distribution the policy was
trained on. Saves:

  - examples/gen_envs_out/env_NN.png      individual cells (full 640x360 wrist)
  - examples/gen_envs_out/contact_sheet.png   all cells tiled + numbered
  - examples/gen_envs_out/dr_params.txt   per-number sampled DR values

The frame saved is the FULL-RESOLUTION 640x360 wrist camera (NOT downsampled),
same as examples/visualize_sim.py, so the lighting/shadows can actually be
judged. The DR (lighting + image pipeline) is identical to training; only the
final 80x144 downsample the policy sees is skipped here.

Tunables via env vars: N_ENVS (per batch), N_BATCHES, SEED, OUT, SETTLE_STEPS.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import utils
import envs  # noqa: F401  registers SO101PlaceCube-v1
import mani_skill.envs  # noqa: F401

# ── Config: matches train_squint.py Args defaults (the "current setup") ──────
N_ENVS      = int(os.environ.get("N_ENVS", 7))      # parallel envs per batch
N_BATCHES   = int(os.environ.get("N_BATCHES", 5))   # resets -> fresh DR samples (7*5=35)
SEED        = int(os.environ.get("SEED", 1))
REDUCE_DR   = int(os.environ.get("REDUCE_DR", 1))   # 1 = reduced (realistic) DR ranges
SETTLE      = int(os.environ.get("SETTLE_STEPS", 2))
OUT         = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "gen_envs_out"))

RENDER_W, RENDER_H = 640, 360     # train render_width / render_height (native wrist)
SHEET_W, SHEET_H   = 320, 180     # per-cell size in the tiled contact sheet only

os.makedirs(OUT, exist_ok=True)

# Reduced ("realistic") DR ranges, re-centered on the envs the user judged real
# (#3, #22 best; #5,#10,#19). Tightens only the multiplicative blow-out/black-out
# knobs — exposure, image gain, gamma, color jitter, noise — and leaves the
# light-intensity structure (key floor, fill/point/ambient) alone. Compare to
# the current training defaults in envs/base_random_env.py RandomizationConfig:
#   exposure_range            (0.25, 2.2)  -> (0.70, 1.40)   # kill 9x span -> ~2x
#   image_channel_gain_range  (0.60, 1.50) -> (0.75, 1.05)   # cap blow-out
#   image_gamma_range         (0.50, 1.70) -> (0.80, 1.25)   # center near 1.0
#   light_color_jitter        (0.70, 1.30) -> (0.85, 1.15)   # less color cast
#   image_noise_sigma_range   (0.00, 0.025)-> (0.00, 0.015)  # less webcam grain
REDUCED = {
    "exposure_range": (0.70, 1.40),
    "image_channel_gain_range": (0.75, 1.05),
    "image_gamma_range": (0.80, 1.25),
    "light_color_jitter": (0.85, 1.15),
    "image_noise_sigma_range": (0.0, 0.015),
}

# Exactly the env_kwargs train_squint.py builds for SO101PlaceCube-v1 with the
# current Args defaults (+ split_2cube_quiet launch: split_only_reward on).
dr_cfg = {"shadows": True, "camera_lag_substeps_range": (0, 0)}
if REDUCE_DR:
    dr_cfg.update(REDUCED)
    print("[REDUCED DR] overrides:", {k: v for k, v in REDUCED.items()})
env_kwargs = dict(
    obs_mode="rgb",
    render_mode="all",
    sim_backend="gpu",
    sensor_configs=dict(width=RENDER_W, height=RENDER_H),
    domain_randomization=True,
    domain_randomization_config=dr_cfg,
    n_distractors=1,
    use_real_bowl=True,
    pick_only_reward=False,
    split_only_reward=True,
    sim_freq=100,
    control_freq=10,
)

print(f"Building SO101PlaceCube-v1 x{N_ENVS} (DR on, shadows on, native {RENDER_W}x{RENDER_H} wrist, no downsample)")
env = gym.make("SO101PlaceCube-v1", num_envs=N_ENVS,
               reconfiguration_freq=None, **env_kwargs)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
# NOTE: no DownsampleObsWrapper — keep the native 640x360 wrist frame so the
# lighting can be judged (see examples/visualize_sim.py downsample_size=None).
base = env.unwrapped


def light_intensity(color):
    return float(np.mean(np.asarray(color)[:3]))


def read_dr(i):
    """Read back the ACTUAL applied DR for sub-scene i (post-reset)."""
    d = {}
    try:
        amb = base.scene.sub_scenes[i].render_system.ambient_light
        d["ambient"] = float(np.mean(np.asarray(amb)[:3]))
    except Exception:
        d["ambient"] = float("nan")
    try:
        dl = base._dir_lights[i]
        d["key"] = light_intensity(dl[0].color)
        d["fill"] = float(np.mean([light_intensity(l.color) for l in dl[1:]])) if len(dl) > 1 else 0.0
    except Exception:
        d["key"] = d["fill"] = float("nan")
    try:
        pl = base._point_lights[i]
        d["point"] = float(np.mean([light_intensity(l.color) for l in pl])) if pl else 0.0
    except Exception:
        d["point"] = float("nan")
    try:
        d["gain"]  = float(base._image_channel_gain[i].mean().item())
        d["gamma"] = float(base._image_gamma[i].item())
        d["noise"] = float(base._image_noise_sigma[i].item())
    except Exception:
        d["gain"] = d["gamma"] = d["noise"] = float("nan")
    return d


cells, params = [], []
n = 0
for b in range(N_BATCHES):
    obs, _ = env.reset(seed=SEED + b * 1000)
    action = np.zeros(env.action_space.shape)
    action[..., -1] = 1.0  # open gripper, neutral
    for _ in range(SETTLE):
        obs, *_ = env.step(action)
    rgb = obs["rgb"]
    if rgb.shape[-1] != 3:
        rgb = rgb[..., :3]
    rgb = rgb.detach().cpu().numpy().astype(np.uint8)  # (N, 360, 640, 3)
    for i in range(rgb.shape[0]):
        n += 1
        cells.append((n, rgb[i]))
        params.append((n, read_dr(i)))

env.close()

# ── Write individual cells + numbered contact sheet ──────────────────────────
disp = []
for num, img in cells:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Individual file: full native 640x360 for judging.
    cv2.imwrite(os.path.join(OUT, f"env_{num:02d}.png"), bgr)
    # Contact-sheet tile: downscale only for the overview grid.
    lab = cv2.resize(bgr, (SHEET_W, SHEET_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(lab, (0, 0), (54, 28), (0, 0, 0), -1)
    cv2.putText(lab, f"#{num}", (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    disp.append(lab)

cols = int(np.ceil(np.sqrt(len(disp))))
rows = int(np.ceil(len(disp) / cols))
ch, cw = disp[0].shape[:2]
sheet = np.zeros((rows * ch, cols * cw, 3), np.uint8)
for k, cell in enumerate(disp):
    r, c = divmod(k, cols)
    sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
cv2.imwrite(os.path.join(OUT, "contact_sheet.png"), sheet)

# ── Write DR params table ────────────────────────────────────────────────────
lines = ["# Per-env sampled DR (read back from scene after reset)",
         "# brightness: ambient/key/fill/point are light intensities; image: gain/gamma/noise",
         f"# {'#':>3}  {'ambient':>8} {'key':>7} {'fill':>6} {'point':>6}  {'gain':>5} {'gamma':>5} {'noise':>5}"]
for num, d in params:
    lines.append(f"  {num:>3}  {d['ambient']:>8.3f} {d['key']:>7.3f} {d['fill']:>6.3f} "
                 f"{d['point']:>6.3f}  {d['gain']:>5.2f} {d['gamma']:>5.2f} {d['noise']:>5.3f}")
with open(os.path.join(OUT, "dr_params.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nWrote {len(cells)} envs to {OUT}/  (contact_sheet.png, env_NN.png, dr_params.txt)")
