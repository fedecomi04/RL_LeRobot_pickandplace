# archive/

Debug scripts, sweep experiments, and working notes that aren't needed to train, evaluate, or deploy the policy, but are kept here for reproducibility of the experimentation that went into the bundled checkpoints. Nothing in `archive/` is imported by code under the repo root, `envs/`, `final_utils/`, `deploy_utils/`, or `scripts/`.

## What's here

### `dev_scripts/`
One-off debug, hardware probe, and visualisation utilities used during development:
- `debug_squint_audit.py`, `debug_squint_replay.py` — replay-style debugging of training rollouts.
- `peek_camera.py`, `probe_cameras.py`, `probe_gripper.py`, `probe_motors.py` — raw hardware probes used during SO-101 bring-up.
- `render_camera_check.py`, `render_shadow_examples.py`, `lineup_palette_camera.py`, `probe_shadow_envelope.py` — sim rendering / lighting / shadow checks used while tuning the realistic-materials env.
- `replay_v1_to_v2.py` — converter for the v1 → v2 RLPD demo schema.
- `camera_live.py`, `cube_color_check.py`, `gripper_probe.py`, `preview_final_palette.py`, `snapshot_initial.py`, `spawn_box_view.py`, `table_mask_live.py`, `eval_episode_plots.py`, `gen_train_envs.py` — assorted previously-in-`examples/` debug scripts.

### `sweep_scripts/`
Domain-randomization sweep harnesses that produced the per-knob tuning plots (cube/bowl emission, brightness, saturation, hue, table albedo, etc.) used while landing on the final realistic-materials environment.

### `working_notes/`
- `DEPLOY.md` — outdated upstream-Squint deploy notes, kept for context.
- `TODOS.md`, `commands.md` — running scratch from the experimentation phase.
