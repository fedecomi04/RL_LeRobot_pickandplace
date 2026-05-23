# TODOs

Captured from /plan-eng-review runs. Each item: what, why, pros, cons, context, blockers.

---

## Operational hygiene

### Investigate VM-B git-reset quirk
- **What:** Find what causes `~/squint` on VM-B (`dizzy-maroon-crocodile`) to reset to `origin/<branch>` between sessions, clobbering uncommitted work.
- **Why:** This is the reason uncommitted WIP files keep disappearing. Workaround (`cp /tmp/*.py ~/squint/`) is annoying and risky.
- **Pros:** Stops a recurring operational paper-cut. Makes future uncommitted experimentation safe on VM-B.
- **Cons:** ~20-min investigation diversion; the immediate pain is gone after the `tom-separating-cubes` commit (see plan 2026-05-21_0511).
- **Context:** Suspected location: `scripts/brev_bootstrap_rtx6000.sh` or `scripts/brev_setup.sh` likely runs `git reset --hard` or `git checkout -f` on session start. Confirmed-affected files: `demo_loader.py`, `train_squint.py`, `envs/place.py`. Captured by D8 in `logs/2026-05-21_0511_PLAN_rlpd-300step-verification.md`.
- **Depends on / blocked by:** none.

---

## Collector ergonomics

### Parameterize `T_*` constants in `scripts/collect_rlpd_demos.py`
- **What:** Move `T_HOME`, `T_MOVE`, `T_DESC`, `T_GRIP`, `T_GRASP_HOLD`, `T_LIFT`, `T_TRANS`, `T_HOVER`, `T_REL` from module-level constants to argparse CLI args (with current values as defaults).
- **Why:** Each demo recipe variant currently requires editing the script or maintaining `/tmp/collect_*_compressed.py` copies. Phase 3 of the current plan (if it fires) needs different `T_*` values.
- **Pros:** Cleaner reproducibility; no more `/tmp` script copies; lets us A/B different timing recipes from the command line.
- **Cons:** ~30 min to add and validate; only valuable if Phase 3 (full recollect) fires.
- **Context:** Current defaults at `scripts/collect_rlpd_demos.py:37-39`. Phase 3 candidate values from handoff §5 Step 2: `T_HOME=10, T_MOVE=50, T_DESC=25, T_GRIP=30, T_GRASP_HOLD=15, T_LIFT=25, T_TRANS=40, T_HOVER=10, T_REL=15`. Captured by D9 in `logs/2026-05-21_0511_PLAN_rlpd-300step-verification.md`.
- **Depends on / blocked by:** Phase 3 trigger (Phase 1 verdict must come back ZERO).
