# Eval commands — checkpoint variants to test

Run from the repo root on the `tom-separating-cubes` branch with `conda activate squint`. Measure the bowl once with `python -m final_utils.teach_bowl_xy` and substitute `0.13 0.16` below.

Goal-colour codes: `0 red · 1 blue · 2 green · 3 yellow · 4 purple · 5 orange`.

## Eval 1 — pick & place (1 cube)

```bash
# savage-DR r3 pick policy (2026-05-20 04:37, step 2.40M, commit 456ef0f)
python -m final_utils.pick_place --goal_color 0 --bowl_xy 0.25 0.2 \
    --checkpoint runs/eval1_place_80x144_savageDR_r3/ckpt.pt
```

## Eval 2 — split 2 cubes + pick & place

Pick policy stays at the bundled `final_utils/pick_place_policy.pt`; only the split policy is swapped.

```bash
# Phase B v2 split policy (2026-05-20 20:19, commit cd04b2b)
python -m final_utils.eval2 --goal_color 0 --bowl_xy 0.25 0 \
    --checkpoint2 runs/split_2cube_quiet_v2_80x144/ckpt.pt
```

## Eval 3 — split 4 cubes + 3 ordered picks

Pick policy stays at the bundled `final_utils/pick_place_policy.pt`; only the 4-cube split policy is swapped. `--goal_colors` is the 3-colour pick order.

```bash
# closest-first v1 (2026-05-21 08:29, commit beed00d) — newest BESTTTTttttttttttttttttttttttttttttttttttT
python -m final_utils.eval3 --goal_colors 1 3 4 --bowl_xy 0.13 -0.16 \
    --checkpoint2 runs/split_4cube_cf_v1_80x144/ckpt.pt



# sequential v1 (2026-05-21 04:42)

```

## Calibration prereqs (auto-loaded; only redo if hardware moved)

- `table_z_calib.json` — `python examples/table_z_calib.py` (slide the closed fingertip across the table)
- `hue_calib.json` — `python -m final_utils.calib_colors` (lay all 6 cubes left → right in palette order)
- Bowl xy — `python -m final_utils.teach_bowl_xy` (live, every run; move tip over bowl centre)

All checkpoint paths above are tracked under `runs/` on this branch — `git pull` is enough, no rsync needed.
