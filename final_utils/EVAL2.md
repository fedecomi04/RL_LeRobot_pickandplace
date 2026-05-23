# Eval 2 — Two-cube Split → Pick & Place

Same task as [Eval 1](EVAL1.md) but the scene has **two cubes** that may be
touching/overlapping. The robot first runs a dedicated **split** policy to push
the cubes apart, returns to the rest pose, then runs the unchanged Eval 1
pick-and-place. Success = the goal-colour cube is grasped, carried to the bowl,
and released.

All commands run from the repo root with the env active:

```bash
conda activate squint
cd /home/team44/squint
```

Hardware + one-time calibration (table height, cube colours, bowl position) are
**identical to Eval 1** — see [EVAL1.md](EVAL1.md). The same `table_z_calib.json`
/ `hue_calib.json` are auto-loaded.

---

## Run the eval

```bash
python -m final_utils.eval2 --goal_color 0 --bowl_xy <x> <y>
```

- `--goal_color`: `0 red · 1 blue · 2 green · 3 yellow · 4 purple · 5 orange`
- `--bowl_xy`: the bowl `x y` (metres, robot base frame). **Default `0.25 0.20`.**
- Both checkpoints are bundled in `final_utils/` and default automatically:
  - `--checkpoint1` — the Eval 1 pick policy (`pick_place_policy.pt`)
  - `--checkpoint2` — the 2-cube split policy (`split_policy.pt`)

Exit code is **0 on success**, **1 if it didn't finish**, so it scripts cleanly:

```bash
python -m final_utils.eval2 --goal_color 0 --bowl_xy 0.25 0.20 && echo "EVAL2 PASS"
```

From Python:

```python
from final_utils import split_pick_place
ok = split_pick_place(goal_color=0, bowl_xy=(0.25, 0.20))
```

`--no-split` skips the split phase, making it behave exactly like Eval 1.

---

## Variants — alternative 2-cube split checkpoints

Override the bundled `final_utils/split_policy.pt` by passing `--checkpoint2 <path>` (the pick policy `--checkpoint1` stays at the bundled `pick_place_policy.pt`). Run from the repo root on the `tom-separating-cubes` branch — all paths below are tracked in git.

| Date trained | Checkpoint | What it is |
|---|---|---|
| 2026-05-20 20:19 | `runs/split_2cube_quiet_v2_80x144/ckpt.pt` | Phase B v2 — quiet-motion penalty (too lenient), commit `cd04b2b` |

```bash
# Phase B v2 split policy
python -m final_utils.eval2 --goal_color 0 --bowl_xy <x> <y> \
    --checkpoint2 runs/split_2cube_quiet_v2_80x144/ckpt.pt
```

---

## What happens in a run

1. **Split** — the split policy (smaller `--split_action_scale 0.3`) nudges the
   two cubes apart. There's no clean "separated" signal, so the phase is timed
   off the arm: it runs until the FK end-effector drops below `--split_below_z`
   (6 cm above the table), then `--split_run_s` (8 s) more, then stops
   (`--split_max_s` 20 s hard cap). Masking here is **background-only** — the
   split policy must see *both* cubes, so the cube-colour mask is OFF.
2. **Return to rest** — the arm goes back to the start pose.
3. **Pick & Place** — the unchanged Eval 1 pipeline: approach → gate → grasp →
   lift → carry to the bowl → release. Full mask stack (background + cube-colour
   mask greying the non-goal cube).

## Useful split flags

| Flag | Default | What |
|------|---------|------|
| `--split` / `--no-split` | on | run the split phase (off = plain Eval 1) |
| `--split_action_scale` | `0.3` | action-scale for the split policy |
| `--split_below_z` | `0.06` | tip height (m) that starts the split-end countdown |
| `--split_run_s` | `8.0` | seconds to keep splitting after dropping below the threshold |
| `--split_max_s` | `20.0` | hard cap if the tip never descends |

The pick/place flags (`--action_scale`, `--place_z`, `--place_speed`, …) and the
hardware overrides (`--robot_port`, `--camera_index`, `--no-viz`) are the same as
Eval 1.

## Saving the run tail to Rerun (`.rrd`)

By default the **last 20 s** of camera frames (raw + masked policy input) and the
joint scalars — across **both** the split and pick phases — are buffered in RAM
and written to a Rerun recording at the end of the run, openable in the
[rerun.io](https://rerun.io) viewer (`rerun eval2_last20s_*.rrd`).

| Flag | Default | What |
|------|---------|------|
| `--save_window` / `--no-save_window` | on | save the run's tail to a `.rrd` |
| `--save_window_s` | `20.0` | length (s) of the saved rolling window |
| `--rrd_path` | auto | output path (default `eval2_last<N>s_<timestamp>.rrd` in cwd) |

## Notes

- `final_utils/eval2.py` is the canonical Eval 2 script. It reuses the exact Eval
  1 grasp+place state machine via `run_pick_place()` in `pick_place.py` (single
  source of truth); it only adds the split phase + two-phase orchestration.
- `infer_eval2_linux.py` at the repo root is a standalone mirror of the same
  pipeline (no `final_utils` import), kept for parity with `infer_linux.py`.
