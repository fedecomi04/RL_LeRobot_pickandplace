#!/usr/bin/env bash
# Phase C — shadow-aware fine-tune of the Phase-B "quiet" split-cube policy.
#
# Warm-starts from runs/split_2cube_quiet_80x144/ckpt.pt (the Phase-B output)
# and runs ~1M steps with directional-light shadows ON. Drops num_envs to
# whatever the shadow-envelope probe (scripts/probe_shadow_envelope.py) found
# fits with shadows=True / 1 directional light at training resolution.
#
# Requires NUM_ENVS_SHADOW to be set explicitly — no default — so this can't
# be launched without running the probe first. That's the deploy-contamination
# guardrail called out in the Phase-C plan: every shadow-FT setting is
# hardcoded here, never silently inherited from environment.
#
# Usage (on the Brev VM, after Phase B has produced its ckpt):
#   NUM_ENVS_SHADOW=256 bash scripts/brev_run_shadow_finetune.sh
set -euo pipefail

# ── Hard prerequisites ──────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-runs/split_2cube_quiet_80x144/ckpt.pt}"
if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: warm-start ckpt missing at $CHECKPOINT" >&2
  echo "  Phase B must finish before Phase C can start." >&2
  exit 1
fi

if [ -z "${NUM_ENVS_SHADOW:-}" ]; then
  echo "ERROR: NUM_ENVS_SHADOW is unset." >&2
  echo "  Run scripts/probe_shadow_envelope.py first to find the largest" >&2
  echo "  num_envs that fits at shadows=True / 1 directional light, then" >&2
  echo "  export NUM_ENVS_SHADOW=<that number>." >&2
  exit 1
fi

# ── Hardcoded Phase-C knobs (DO NOT inherit from environment) ───────────────
# This launcher is single-purpose — every value below is what makes "shadow
# fine-tune of the quiet split policy" specifically. Inheriting these from
# the calling shell would risk contaminating other training/deploy paths.
export EXP_NAME="split_2cube_quiet_shadow_80x144"
export SHADOWS=true
export NUM_DIRECTIONAL_LIGHTS=1     # only value that fits at scale with shadows on
export TOTAL_TIMESTEPS=1000000      # fine-tune length, not full retrain
export POLICY_LR=1e-4               # warm-start convention: 1/3 of default 3e-4
export Q_LR=1e-4
export NUM_ENVS="$NUM_ENVS_SHADOW"  # set explicitly by caller; no default

# Inherit Phase-B reward shaping (gripper-quiet, action-smooth, anti-fling,
# no hover, 2.5cm target gap, 5cm fling threshold). These are the constraints
# Phase B already trained the policy under — Phase C only ADDS shadow
# robustness, doesn't change the task.
export SPLIT_TARGET_GAP="${SPLIT_TARGET_GAP:-0.025}"
export SPLIT_HOVER="${SPLIT_HOVER:-false}"
export SPLIT_FAR_PENALTY_COEF="${SPLIT_FAR_PENALTY_COEF:-5.0}"
export SPLIT_FAR_PENALTY_DIST="${SPLIT_FAR_PENALTY_DIST:-0.05}"
export SPLIT_GRIPPER_ACTION_PENALTY_COEF="${SPLIT_GRIPPER_ACTION_PENALTY_COEF:-1.0}"
export ACTION_SMOOTH_COEF="${ACTION_SMOOTH_COEF:-2.0}"

# Distinct wandb group so Phase-C runs cluster together.
# Phase-C runs go to Tommaso's personal wandb account (tom-gazzini-ethrc),
# NOT the fedecominelli04_robot account that hosted Phase A/B. The API key
# is in ~/.netrc on the VM (chmod 600); wandb picks it up automatically.
export WANDB_ENTITY="${WANDB_ENTITY:-tom-gazzini-ethrc}"
export WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
export WANDB_GROUP="${WANDB_GROUP:-SQUINT-SHADOW-FT-$(date +%Y%m%d-%H%M)}"

# Buffer / batch / updates inherit from brev_run_split.sh defaults (500K /
# 512 / 256) — those are sized for the 80x144 / RTX 6000 envelope and don't
# need to change for the shadow phase.

# Hand off to the shared launcher, which already plumbs every knob above.
CHECKPOINT="$CHECKPOINT" bash "$(dirname "$0")/brev_run_split.sh"
