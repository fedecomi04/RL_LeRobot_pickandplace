# RLPD Synthetic Demo Pipeline

Generates scripted pick-and-place demonstrations for `SO101PlaceCube-v1` in
ManiSkill simulation, in the HDF5 format `train_squint.py`'s RLPD loader
consumes. ~7 minutes for 50 demos, balanced across 6 cube colors.

The pipeline replaces hours of teleoperated data collection: a forward-kinematics
IK planner produces the trajectory, the simulator executes it, and we record the
observations and actions that `train_squint.py` would see if a real policy had
performed the task.

---

## TL;DR

```bash
# On the Brev VM, in ~/squint:
python scripts/collect_rlpd_demos.py
# → /tmp/rlpd_50demos/demos.h5            (102 MB, 50 demos)
# → /tmp/rlpd_50demos/sanity_<color>.mp4  (6 videos, 1 per color, ~550 KB each)
# → /tmp/rlpd_50demos/meta.json           (summary)

# Validate the file structurally:
python scripts/check_rlpd_demos_v2.py /tmp/rlpd_50demos/demos.h5
# → "ALL CHECKS PASSED" plus 2 expected deviation warnings (see §6).
```

The h5 is consumed by `demo_loader.load_demo_buffer()` and dropped into the
RLPD replay buffer in `train_squint.py`.

---

## 1. Why scripted demos at all

RLPD bootstraps an online RL policy from offline expert trajectories. For
`SO101PlaceCube-v1`, the natural source would be teleoperated pickups on the
real arm — slow, error-prone, single-color. With a working forward-kinematics
model (`so101_fk.py`) and ManiSkill's GPU-batched sim, we can synthesize
hundreds of varied demos per hour entirely in simulation, matching the obs/
action format the policy will see at training time.

A demo here = one continuous successful trajectory: arm starts at the home
pose → reaches above the cube → descends → closes the gripper → lifts → moves
above the bowl → releases.

---

## 2. Working recipe (the trajectory)

All phase step counts (`T_*`) are CLI args on `collect_rlpd_demos.py` (see
`--help`); defaults below are the "medium" recipe locked in plan
`logs/2026-05-21_0545_PLAN_regenerate-demos-medium.md`. The original 785-step
recipe is still reproducible via the per-flag values noted in the rightmost
column.

Per demo, the IK planner produces ~270 control steps (= ~9 s @ 30 Hz render,
~27 s @ 10 Hz control), structured as six cosine-interpolated phases:

| Phase | Default (lerp+hold) | Gripper state | Original 785-recipe |
|---|---|---|---|
| Home → pre-grasp (above cube) | 55 + 10 | open (120°) | 90 + 30 (`--t_move=90 --t_home=30`) |
| Pre-grasp → descend | 30 + 25 | partial open (60°) | 65 + 25 (`--t_desc=65`) |
| Close gripper | 25 + 12 | close (60° → 5°) | 100 + 100 (`--t_grip=100 --t_grasp_hold=100`) |
| Lift (single-joint, preserves wrist) | 35 + 10 | closed (5°) | 110 + 30 (`--t_lift=110 --t_home=30`) |
| Transit to bowl | 55 + 10 | closed (5°) | 120 + 35 (`--t_trans=120 --t_hover=35`) |
| Open gripper (release) | 18 + 10 | open (120°) | 50 + 30 (`--t_rel=50 --t_home=30`) |

**Bound discipline.** Half-cosine peak delta = `|Δq|·π/(2·n_lerp_steps)` must
stay ≤ 0.05 rad/step (arm) / 0.20 rad/step (gripper). The medium defaults are
sized for a 1.5-rad worst-case shoulder reach (`T_MOVE=T_TRANS=55` →
`1.5·π/110 = 0.043`, 86% of bound — 14% headroom). The collector applies a
strict pre-write check (`--bound_max_normalized=1.0` default, tighter than
the loader's 1.05 cap) and silently drops demos that exceed it; the batch
loop retries with new cube spawns.

Key IK choices that make the grasp robust:

- **Face-to-face wrist alignment.** After position IK, rotate `wrist_roll` so
  the closed jaw line is aligned with one of the cube's four face normals
  (cube has 4-fold Z symmetry). Reduces the inclination between gripper and
  cube faces from ~27° to <1°.
- **Single-joint lift.** The lift uses `shoulder_lift -= 0.55, elbow_flex +=
  0.20` and leaves `wrist_flex` / `wrist_roll` alone. Preserving the wrist
  orientation prevents the cube from tumbling out during the lift acceleration.
- **Grasp target slightly below cube center.** Set the target TCP to
  `(cube_x, cube_y, cube_z - 0.003 m)` so the fingers wrap *under* the cube
  edge, not on top of it.

---

## 3. The env config that works (and the three that don't)

We spent significant time discovering that the env stack the original v2
handoff spec required (`logs/2026-05-20_HANDOFF_rlpd-demos-v2-spec.md`) is
*incompatible with scripted IK*. A/B tests pin down each culprit:

| Env config (otherwise identical, same seed) | Max cube z lifted | Cube in bowl? |
|---|---|---|
| V3 base — `gym.make(...)` only | **10.4 cm** | **✓ (0.4 cm dist)** |
| V3 base + `ManiSkillVectorEnv` wrapper | 1.0 cm | ✗ (never lifts) |
| V3 base + `pd_joint_target_delta_pos` controller | 1.0 cm | ✗ (controller target accumulator corrupts under collision) |
| V3 base + `domain_randomization=True` (default DR config) | ~1.5 cm | ✗ (item friction randomized down to 0.3 breaks grip) |

The collector therefore uses these settings (which all deviate from the spec
but are required for grasps to actually happen):

```python
env = gym.make(
    'SO101PlaceCube-v1',
    num_envs=8,                                # batched for speed
    obs_mode='rgb',
    render_mode='all',
    sim_backend='gpu',
    sensor_configs=dict(width=640, height=360),
    domain_randomization=False,                # not True
    n_distractors=0,
    use_real_bowl=True,
    control_mode='pd_joint_pos',               # not pd_joint_target_delta_pos
    sim_freq=100, control_freq=10,
    pick_only_reward=False, split_only_reward=False,
    action_smooth_coef=0.0,
)
# NO ManiSkillVectorEnv wrapper. NO obs flatteners. Direct gym.make.
```

The three deviations are documented in HDF5 file attrs (`collector_deviation_*`)
so the loader can detect and handle them without reading docs.

---

## 4. Variation strategy

| Knob | Source | Result |
|---|---|---|
| Cube position | env's natural spawn (varies per-seed, per-env-in-batch) | 50 distinct positions across the spawn arc |
| Bowl position | env's natural spawn | 50 distinct positions across the placement region |
| Cube color | post-reset override of `goal_color_idx` + `_set_actor_palette_color` | Balanced across 6 colors (red/blue/green/yellow/purple/orange), ~9 demos per color |
| Cube orientation | random Z rotation per env (env default) | Wrist-roll alignment handles each via the 4-fold symmetry |

To rebalance colors automatically: each new batch picks the colors that
currently have the fewest demos.

---

## 5. Output format (v2 HDF5 schema)

```
file attrs:
  format_version    '2.0'
  env_id            'SO101PlaceCube-v1'
  control_mode      'pd_joint_pos'                       ← deviation
  n_distractors     0
  use_real_bowl     True
  domain_randomization  False                            ← deviation
  apply_jitter      True
  rgb_h, rgb_w      80, 144
  state_dim         21
  action_dim        6
  arm_delta_max     0.05
  grip_delta_max    0.20
  num_demos         50
  num_colors        6
  T                 ~270 (medium recipe; 785 in the original 50-demo run)
  reward_v_min      -20.0
  reward_v_max      20.0
  collector_commit  <git rev-parse --short HEAD>
  collected_at_utc  <ISO 8601>
  state_layout      'bowl_xyz_robot_frame(3) + target_qpos(6, synthesized=action[t-1]) + goal_color_onehot(6) + noisy_qpos(6)'
  collector_deviation_control_mode  '<explanatory text>'
  collector_deviation_msvecenv      '<explanatory text>'
  collector_deviation_dr            '<explanatory text>'

per-demo group demo_NNN:
  obs/rgb     (785, 80, 144, 3) uint8   gzip-4   — area-resized from 640×360
  obs/state   (785, 21)         float32 gzip-4   — see state_layout attr
  actions     (785, 6)          float32 gzip-4   — ABSOLUTE joint targets in rad
  rewards     (785,)            float32
  terminals   (785,)            bool
  attrs:
    color_idx   int          # 0..5
    cube_pos    (3,)         # initial cube xyz, world frame
    bowl_pos    (3,)         # initial bowl xyz, world frame
    seed        int          # env reset seed
    success     bool         # cube landed in bowl
    return_sum  float        # sum of rewards across trajectory
```

---

## 6. Loader-side conversions

The training env exposes a normalized delta-pos action space (`[-1, 1]^6`,
internally scaled to `[-0.05]*5 + [-0.2]` rad). Our demos are absolute targets.
Two-line conversion at load time:

```python
# actions: (T, 6) float32 — absolute joint targets in rad (read from h5)
deltas = np.empty_like(actions)
deltas[0] = actions[0] - QPOS_START      # QPOS_START hard-coded; or use first qpos
deltas[1:] = actions[1:] - actions[:-1]
deltas = np.clip(deltas, -DELTA_BOUND, DELTA_BOUND)   # DELTA_BOUND = [0.05]*5 + [0.2]
normalized_actions = deltas / DELTA_BOUND              # what env.step() expects
```

State is already the correct 21-d layout (matches training delta-pos
`controller.get_state()` insertion order).

No conversion needed for `obs/rgb`, `rewards`, `terminals`.

---

## 7. Files

| Path | Purpose |
|---|---|
| `scripts/collect_rlpd_demos.py` | Main collector. ~7 min for 50 demos. Saves h5 + 6 sanity videos + meta.json. |
| `scripts/check_rlpd_demos_v2.py` | Structural validation: file attrs, dtypes, shapes, action bounds, reward range, per-demo metadata. Run after collection. |
| `demo_loader.py` | (Teammate's.) Reads the v2 h5 and emits a `TensorDict` for the RLPD replay buffer. |
| `so101_fk.py` | (Existing.) Pure-numpy FK + damped-LS IK. Verified vs SAPIEN to <1 mm. |
| `envs/place.py` | (Existing.) `SO101PlaceCube-v1` definition + `_set_actor_palette_color` (used for color override). |
| `docs/RLPD_DEMO_PIPELINE.md` | This document. |
| `logs/2026-05-20_HANDOFF_rlpd-demos-v2-spec.md` | Original v2 spec (predecessor; superseded by this doc on the deviation points). |

Sanity videos and h5 outputs are gitignored (`logs/`, `*.mp4`). Pull from
the brev VM at `/tmp/rlpd_50demos/` after a run, or just re-run the collector
(~7 min) to regenerate.

---

## 8. Known limitations

1. **~30% IK grasp rate per seed.** The collector compensates with `MAX_BATCHES`
   and per-color quotas; 50 demos × 6 colors finishes in ~10 batches. For more
   demos, increase `NUM_DEMOS_TARGET` and `MAX_BATCHES`.
2. **DR is off.** Lighting/shadow/material randomization that the policy will
   see at training time is NOT present in the recorded RGB frames. Apply
   `ColorJitterWrapper`-style augmentation at training-load time, not collection
   time.
3. **`pd_joint_target_delta_pos` controller bug not fixed.** We worked around it
   by using `pd_joint_pos`. If we ever fix the underlying SAPIEN controller bug,
   the collector and the loader-side conversion can both drop the deviation.
4. **Eval 2 / Eval 3 (multi-cube with distractors) not supported yet.** The IK
   planner currently has no push-distractor phase. Defer until Eval 1 RLPD
   training proves the pipeline integrates.

---

## 9. Validation

The 50-demo file passes 16/18 checks in `check_rlpd_demos_v2.py`. The 2 failing
checks are the documented deviations:

- ✗ `control_mode == pd_joint_target_delta_pos` → ours says `pd_joint_pos`
- ✗ `actions inside env action space` → ours are in joint-limit range (`[-1.92, 2.84]`), not `[-1, 1]` normalized

Both are flagged in `collector_deviation_*` attrs and the loader handles them
per §6.

---

## 10. Next steps

- [ ] Plug into `train_squint.py` RLPD loop and verify the demos accelerate
      learning on Eval 1 vs from-scratch RL.
- [ ] Extend to Eval 2 (1 distractor, face-to-face). The IK planner needs a
      "push distractor aside" phase before the grasp.
- [ ] Investigate the `pd_joint_target_delta_pos` controller's target
      accumulator bug under cube/finger collision (root-cause fix would let us
      drop the deviation).
- [ ] Add DR back into the collector once we have a more robust grasp pipeline
      (or accept the load-time augmentation route).
