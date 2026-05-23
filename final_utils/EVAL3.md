# Eval 3 — Four cubes, ordered colour pick-and-place

Like [Eval 2](EVAL2.md) but the scene has **four cubes**, and the task queries
**three colours** that must be picked and dropped into the bowl **in that exact
order**. The robot first splits the cubes apart (4-cube split policy), then runs
the Eval 1 pick-and-place once per colour, in sequence. Success = all three
cubes placed, in order.

All commands run from the repo root with the env active:

```bash
conda activate squint
cd /home/team44/squint
```

Hardware + one-time calibration (table height, cube colours, bowl position) are
**identical to Eval 1** — see [EVAL1.md](EVAL1.md).

---

## Run the eval

```bash
python -m final_utils.eval3 --goal_colors <c1> <c2> <c3> --bowl_xy <x> <y>
```

- `--goal_colors`: three colours **in pick order** —
  `0 red · 1 blue · 2 green · 3 yellow · 4 purple · 5 orange`
- `--bowl_xy`: bowl `x y` (metres, robot base frame). **Default `0.25 0.20`.**
- Both checkpoints are bundled in `final_utils/` and default automatically:
  - `--checkpoint1` — the Eval 1 pick policy (`pick_place_policy.pt`)
  - `--checkpoint2` — the 4-cube split policy (`split4_policy.pt`)

Example (pick red, then green, then purple):

```bash
python -m final_utils.eval3 --goal_colors 0 2 4 --bowl_xy 0.25 0.20 && echo "EVAL3 PASS"
```

From Python:

```python
from final_utils import split_pick_place_sequence
ok = split_pick_place_sequence(goal_colors=(0, 2, 4), bowl_xy=(0.25, 0.20))
```

Exit code is **0** only if all three cubes were placed in order, else **1**.

---

## Variants — alternative 4-cube split checkpoints

Override the bundled `final_utils/split4_policy.pt` by passing `--checkpoint2 <path>` (the pick policy `--checkpoint1` stays at the bundled `pick_place_policy.pt`). Run from the repo root on the `tom-separating-cubes` branch — all paths below are tracked in git.

| Date trained | Checkpoint | What it is |
|---|---|---|
| 2026-05-21 08:29 | `runs/split_4cube_cf_v1_80x144/ckpt.pt` | closest-first v1 — active pushing, fling tail (47% no-fling, 1.7cm p90 gap), commit `beed00d` |
| 2026-05-21 06:39 | `runs/split_4cube_seq_v2_80x144/ckpt.pt` | sequential v2 |
| 2026-05-21 04:42 | `runs/split_4cube_seq_80x144/ckpt.pt` | sequential v1 |

```bash
# closest-first v1 (newest)
python -m final_utils.eval3 --goal_colors 0 2 4 --bowl_xy <x> <y> \
    --checkpoint2 runs/split_4cube_cf_v1_80x144/ckpt.pt

# sequential v2
python -m final_utils.eval3 --goal_colors 0 2 4 --bowl_xy <x> <y> \
    --checkpoint2 runs/split_4cube_seq_v2_80x144/ckpt.pt

# sequential v1
python -m final_utils.eval3 --goal_colors 0 2 4 --bowl_xy <x> <y> \
    --checkpoint2 runs/split_4cube_seq_80x144/ckpt.pt
```

---

## What happens in a run

1. **Split** — the 4-cube split policy nudges the cubes apart, timed with the
   **same heuristic as Eval 2**: run until the FK tip drops below
   `--split_below_z` (6 cm), then `--split_run_s` (8 s) more, then stop. Masking
   is background-only.
2. **Return to rest.**
3. **For each colour, in order** — run the Eval 1 pick-and-place (approach →
   gate → grasp → lift → carry to bowl → release). If the cube isn't placed, the
   whole pick-and-place is **retried** (up to `--max_attempts_per_color`,
   default 5; `<=0` = unlimited). The arm returns to rest between every attempt
   and every colour, then moves on to the next colour. The sequence aborts if a
   colour can't be placed (order matters).

## Useful flags

| Flag | Default | What |
|------|---------|------|
| `--max_attempts_per_color` | `5` | retries per colour before giving up (`<=0` = unlimited) |
| `--split` / `--no-split` | on | run the 4-cube split first |
| `--split_action_scale` | `0.3` | action-scale for the split policy |
| `--split_run_s` | `8.0` | seconds to keep splitting after dropping below the threshold |
| `--distractor_mask` | on | grey non-goal cubes during each pick (needed to target the right colour among 4) |
| `--save_window` / `--save_window_s` | on / `20` | save the run's tail to a Rerun `.rrd` |

The pick/place flags (`--action_scale`, `--place_z`, `--place_speed`, …) and the
hardware overrides (`--robot_port`, `--camera_index`, `--no-viz`) are the same as
Eval 1/2.

## Notes

- `final_utils/eval3.py` reuses the Eval 1 grasp+place state machine
  (`run_pick_place`) and the Eval 2 split phase (`run_split`) — single source of
  truth; it only adds the ordered three-colour loop.
- The cube-colour (distractor) mask is **ON** here (unlike single-cube Eval 1),
  since with four cubes in view it's what steers the colour-blind pick policy to
  the correct cube.
