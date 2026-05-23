# Eval 1 — Pick & Place

Pick up the cube of a queried colour and drop it into the bowl. The vision RL
policy aligns to the cube; a hardcoded FK/IK layer does the precise grasp, carry,
and release. Success = the cube is grasped, carried to the bowl, and released.

All commands run from the repo root with the env active:

```bash
conda activate squint
cd /home/team44/squint
```

Hardware: SO101 follower on `/dev/ttyACM0` + wrist camera on `/dev/video0`
(pass `--robot_port` / `--camera_index` if different).

---

## One-time calibration (per rig / per setup)

These write small JSON files at the repo root that the eval **auto-loads**. Redo
them if you move the camera, change the lighting, or change the table.

### 1. Table height vs. reach — `table_z_calib.json`

The real arm's geometry drifts from the model with extension, so the FK height
that means "touching the table" changes with reach. Calibrate it:

```bash
python examples/table_z_calib.py
```

The arm goes limp (gripper held closed). **Slide the closed fingertip across the
table from near the base out to full reach** (and back), then `Ctrl+C`. It fits
`z_table(r)` and saves it. Without this file the eval assumes a flat table and
far cubes won't be grasped well.

### 2. Cube colours — `cube_color_calib.json` (MAIN colour filter)

The mask keeps the goal cube + gripper and greys everything else, and it picks
the goal cube **by colour**. The classifier compares each segment's mean **RGB
chromaticity** (brightness-normalised R:G:B ratios) to a per-cube reference —
this separates red/orange/yellow cleanly, where HSV hue collapsed them. Those
references come from this calibration, measured **one cube at a time** so there's
no left/right ambiguity:

```bash
python -m final_utils.calib_cube_colors_live --camera_index 1
```

> Put **one** cube on the table (the gripper may be in view — it's auto-skipped).
> The detected cube is outlined and shows its live chromaticity + current colour.
> Press the colour's key to capture it:
>
> ```
> 0 red   1 blue   2 green   3 yellow   4 purple   5 orange
> ```
>
> Top bar shows what's captured (`redOK blu-- …`). Do all six, press **`s`** to
> save `cube_color_calib.json`, `q` to quit (`r` resets).

`cube_color_calib.json` is **auto-loaded as the primary colour reference** for the
mask (it takes priority over `hue_calib.json`). Because the references are the
cubes' *real* captured colours, detection is robust and red no longer drifts to
orange. Re-run it whenever you change the lighting/camera.

Check it live — this labels every segment with the colour it classifies as plus
`d` (chromaticity distance) and `c` (colourful fraction); the goal cube should
read its own colour at small `d`:

```bash
python -m final_utils.fastsam_live --goal_color 2 --camera_index 1
```

> **Legacy (HSV) calibration — `hue_calib.json`.** The older tool lays all 6 cubes
> in a row LEFT→RIGHT (`red blue green yellow purple orange`) and saves hues:
> `python -m final_utils.calib_colors`. Only a fallback when
> `cube_color_calib.json` is absent — prefer the per-cube tool above.

### 3. Bowl position — taught live (per bowl placement)

The place step flies to an absolute `(x, y)` in the **robot base frame** (FK
frame), which is *not* a frame you can eyeball. Teach it:

```bash
python -m final_utils.teach_bowl_xy
```

The arm goes limp (gripper closed). **Move the gripper tip directly over the bowl
centre**, then `Ctrl+C`. It prints the base-frame `x y` to pass to the run.

---

## Run the eval

```bash
python -m final_utils.pick_place --goal_color 0 --bowl_xy <x> <y>
```

- `--goal_color`: `0 red · 1 blue · 2 green · 3 yellow · 4 purple · 5 orange`
- `--bowl_xy`: the `x y` from step 3 (metres, robot base frame).
- Defaults already set: `--action_scale 0.15`, `--episode_steps 1000`, bundled
  checkpoint `final_utils/pick_place_policy.pt`.

Exit code is **0 on success** (cube placed) and **1 if it didn't finish**, so it
scripts cleanly:

```bash
python -m final_utils.pick_place --goal_color 0 --bowl_xy 0.18 -0.06 && echo "EVAL1 PASS"
```

From Python:

```python
from final_utils import pick_and_place
ok = pick_and_place(goal_color=0, bowl_xy=(0.18, -0.06))
```

---

## Variants — alternative pick checkpoints

Override the bundled `final_utils/pick_place_policy.pt` by passing `--checkpoint <path>`. Run from the repo root on the `tom-separating-cubes` branch — all paths below are tracked in git.

| Date trained | Checkpoint | What it is |
|---|---|---|
| 2026-05-20 04:37 | `runs/eval1_place_80x144_savageDR_r3/ckpt.pt` | full pick→place + savage-DR (r3) @ step 2.40M, commit `456ef0f` |

```bash
# savage-DR r3 pick policy
python -m final_utils.pick_place --goal_color 0 --bowl_xy <x> <y> \
    --checkpoint runs/eval1_place_80x144_savageDR_r3/ckpt.pt
```

---

## What happens in a run

1. **Approach** — policy drives the arm and gripper toward the cube.
2. **Gate** — fires when the descent stalls near the table (or hits the height
   gate). The Rerun viewer shows the masked policy input (`camera/policy_input`)
   so you can confirm the bowl/other cubes are greyed and the goal cube is kept.
3. **Grasp** — nudge to centre, full close, verify by the gripper stall angle,
   hold, lift 5 cm. A miss retreats and retries.
4. **Place** — IK to the bowl `(x, y)` at 10 cm above the table, hold 0.5 s, open
   to drop. → success.

## FastSAM masking (alternative to the colour mask)

Instead of the HSV table/distractor mask, the policy input can be built with
**FastSAM** — a "segment everything" model. Per frame it segments all objects,
keeps the segment whose interior matches the queried colour (the goal cube, dark
side faces and all) plus the baked gripper silhouette, and paints everything else
a constant grey. The FastSAM inference time is **printed every control step**:

```bash
python -m final_utils.pick_place --goal_color 1 --bowl_xy <x> <y> --fastsam
# [fastsam]   5.2ms  segs= 14  cube gfrac=0.96 area=701
```

- `--fastsam_weights` `FastSAM-s.pt` (default, ~5 ms on an RTX 5070) or
  `FastSAM-x.pt` (~14 ms, more accurate on small/shadowed cubes).
- `--fastsam_imgsz 640`, `--fastsam_grey 128` (the constant fill value).
- Weights live in `weights/`; first use warms the model up (~40 ms one-off).
- **Missed detection → holds the previous frame's mask** (so the cube doesn't
  flicker to grey when a frame drops it, usually exposure washout). Reused for up
  to `FASTSAM_MAX_HOLD_FRAMES` (30) consecutive misses, then it gives up. The
  print shows `MISS → held prev mask (k/30)`. Disable via `il.FASTSAM_HOLD_ON_MISS`.
- **Run `calib_cube_colors_live` first** (step 2). FastSAM picks the cube *by
  colour*, so without real references warm lighting makes red read as orange and
  cubes get missed (then the mask greys everything but the gripper). The per-cube
  `cube_color_calib.json` is the main colour filter and fixes this.
- **Gripper** is kept straight from the segments (the dark, bottom-anchored one,
  any jaw angle) via `--fastsam_gripper sam` (default); non-goal cube-coloured
  segments are greyed on top, the white/grey bowl is always greyed.

Try it live without the robot (camera-only, shows raw | masked side-by-side):

```bash
python -m final_utils.fastsam_live --goal_color 1            # live window
python -m final_utils.fastsam_live --goal_color 1 --probe 30 --no-window  # headless, saves overlays
```

## Tuning the grasp heuristic (wait times + thresholds)

The pick is RL-approach → **FK-gated hardcoded grasp**: gate (stop the approach) →
wait → nudge → close → verify → hold → lift, with back-off+retry on a miss. Every
wait time and threshold in that state machine is a CLI flag — defaults live in
`infer_linux.py`; pass a flag to override just that one for the run (it prints
`Grasp-heuristic overrides: …` at start). `python -m final_utils.pick_place --help`
lists them all with the current default in parentheses.

| Flag | Default | What |
|------|---------|------|
| **Gate** — when the approach stops and the grasp begins | | |
| `--gate_z` | `0.004` | gate height above the table (m); fires the grasp |
| `--gate_z_slope` | `0.0` | extra gate height per m of reach; raise if **far** cubes never trigger |
| `--gate_stall` / `--no-gate_stall` | on | also fire when the descent plateaus |
| `--stall_s` | `0.4` | s of no-further-descent before the stall gate fires |
| `--stall_eps` | `0.003` | min descent (m) that still counts as descending |
| `--engage_z` | `0.06` | stall gate only armed within this height of the table (m) |
| **Close** — the corrective nudge + the grip | | |
| `--grasp_wait_s` | `0.0` | s of extra policy run after the gate before closing |
| `--nudge_m` | `0.01` | corrective back-shift toward base before closing (m) |
| `--nudge_z` | `-0.01` | tip height vs table for the nudge (m; −ve = press in). Raise toward 0 if it digs in |
| `--nudge_settle_s` | `0.2` | s to settle into the nudged pose before closing |
| `--close_deg` | `-10.0` | commanded full-close gripper angle (sim deg) |
| `--close_s` | `0.4` | s for the gripper to close + settle, then decide hit/miss immediately |
| `--empty_below_deg` | `-5.0` | grasp-success threshold: gripper angle > this = grasped |
| **Miss** — back off + retry | | |
| `--max_retries` | `3` | retry attempts before giving up |
| `--retreat_up_m` / `--retreat_back_m` | `0.10` / `0.05` | how far the tip backs off on a miss (m) |
| `--retreat_speed` | `0.60` | tip speed through the back-off (m/s) |
| **Hold + lift** | | |
| `--hold_s` | `0.0` | s to hold the closed grasp before lifting |
| `--lift_m` / `--lift_s` | `0.08` / `1.0` | how far / how long to raise the cube after a grasp |

Example — give the policy longer to settle, a slightly looser gate, and a hold:

```bash
python -m final_utils.pick_place --goal_color 1 --bowl_xy 0.18 -0.06 \
    --grasp_wait_s 0.8 --gate_z 0.006 --nudge_settle_s 0.7 --close_s 1.3 --hold_s 0.3
```

## Useful flags

| Flag | Default | What |
|------|---------|------|
| `--fastsam` / `--no-fastsam` | **on** | FastSAM cube+gripper keep mask (prints SAM ms/step) |
| `--arm_vel` / `--gripper_vel` | `3.0` / `9.0` | **overall speed** envelope (rad/s); raise for a faster approach |
| `--action_scale` | `0.15` | scales the policy action (higher = faster/more aggressive approach) |
| `--episode_steps` | `5000` | max control steps before giving up |
| `--place_z` | `0.10` | drop height above the table (m) |
| `--place_open_wait_s` | `0.5` | hold over bowl before opening |
| `--place_speed` | `1.30` | carry speed to the bowl (m/s) |
| `--place_xy_tol` | `0.015` | xy tolerance to count as "over the bowl" (m) |
| `--no-viz` | (on) | disable the Rerun viewer |
| `--robot_port` / `--camera_index` | auto-detect | hardware overrides (rarely needed) |

> **Robot port and camera index are auto-detected** — you normally don't pass
> either. Both the SO101 serial port and the wrist cam re-enumerate
> (`/dev/ttyACM0↔1`, `/dev/video0↔1`) on USB resets, so each is resolved via its
> stable `/dev/serial/by-id/…` / `/dev/v4l/by-id/…-video-index0` symlink. The
> camera also **auto-reconnects** if unplugged/replugged mid-run. Pass
> `--robot_port <path>` / `--camera_index <n>` only to force a specific device.

## Troubleshooting

- **Doesn't grasp / closes empty** — check the live `above_table` print and the
  masked viewer; re-run table-z calibration if the height looks wrong. If it
  closes *just* before reaching the cube, raise `--nudge_settle_s` / `--close_s`.
- **Drops in the wrong spot** — `bowl_xy` frame is wrong; re-teach with
  `teach_bowl_xy` (don't eyeball the numbers).
- **Bowl/other cube still distracts** — re-run `calib_colors`; tune
  `DISTRACTOR_SAT_MIN` (what counts as coloured) or `GOAL_HUE_TOL` (hue kept
  around the goal) in `infer_linux.py`.
