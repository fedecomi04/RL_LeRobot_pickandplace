"""IK demo loader for RLPD on SO101PlaceCube.

Reads a v2-schema HDF5 file (see logs/2026-05-20_HANDOFF_rlpd-demos-v2-spec.md)
and returns a TensorDict that drops into the existing replay buffer:

    demo_rb = ReplayBuffer(storage=LazyTensorStorage(args.demo_buffer_size, device=device))
    demo_td = load_demo_buffer(args.demo_file, env=envs, device=device)
    demo_rb.extend(demo_td)

The loader's contract is strict: if the file's state_dim, action bounds, or RGB
shape disagree with the live env, it raises with a one-line message. No silent
translation. Mismatches mean the demo file needs to be re-collected against the
current env config, not patched in software.

The dones field is overridden to False everywhere to match the trainer's
`bootstrap_at_done='always'` convention (see train_squint.py:1208-1213).
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import torch
from tensordict import TensorDict


REQUIRED_FILE_ATTRS = (
    "format_version", "env_id", "control_mode",
    "rgb_h", "rgb_w", "state_dim", "action_dim",
    "arm_delta_max", "grip_delta_max", "num_demos",
)


def _get_env_shapes(env) -> dict:
    """Probe the live (wrapped) env for the canonical obs/action shapes the
    replay buffer expects.

    Reads from the BATCHED observation_space (which gym.ObservationWrapper
    keeps in sync) rather than single_observation_space (which some wrappers
    forget to update). Slices off the leading num_envs dim. Falls back to
    single_observation_space if the batched space isn't available."""
    base = env.unwrapped
    obs_space = env.observation_space
    if "rgb" in obs_space.spaces:
        batched_rgb_shape = tuple(obs_space["rgb"].shape)
        # ManiSkillVectorEnv reports shape (num_envs, H, W, 3); strip the lead.
        if len(batched_rgb_shape) == 4:
            rgb_shape = batched_rgb_shape[1:]
        else:
            rgb_shape = batched_rgb_shape
        batched_state_shape = tuple(obs_space["state"].shape)
        if len(batched_state_shape) >= 2:
            state_dim = int(np.prod(batched_state_shape[1:]))
        else:
            state_dim = int(np.prod(batched_state_shape))
    else:
        rgb_shape = tuple(env.single_observation_space["rgb"].shape)
        state_dim = int(np.prod(env.single_observation_space["state"].shape))
    act_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    act_high = np.asarray(env.single_action_space.high, dtype=np.float32)
    return dict(rgb_shape=rgb_shape, state_dim=state_dim,
                act_low=act_low, act_high=act_high)


def _validate_against_env(h5: h5py.File, env_shapes: dict) -> None:
    """Hard fail with one-line context on any mismatch."""
    # File-level required attrs
    missing = [k for k in REQUIRED_FILE_ATTRS if k not in h5.attrs]
    if missing:
        raise ValueError(f"demo file missing required attrs: {missing}")

    fmt = str(h5.attrs["format_version"])
    if fmt != "2.0":
        raise ValueError(
            f"demo file format_version={fmt!r}, loader expects '2.0'. "
            "Re-collect against the v2 spec at "
            "logs/2026-05-20_HANDOFF_rlpd-demos-v2-spec.md."
        )

    ctrl = str(h5.attrs["control_mode"])
    if ctrl == "pd_joint_target_delta_pos":
        pass  # native — no conversion needed
    elif ctrl == "pd_joint_pos" and "collector_deviation_control_mode" in h5.attrs:
        # Explicit documented deviation: demos saved with absolute joint targets,
        # loader converts to deltas+normalized at load time. The deviation attr
        # exists so the loader knows the collector knew. Without it, fail.
        pass  # handled in load_demo_buffer below
    else:
        raise ValueError(
            f"demo control_mode={ctrl!r}, loader expects "
            "'pd_joint_target_delta_pos' (or 'pd_joint_pos' with "
            "collector_deviation_control_mode attr explicitly documenting the "
            "absolute→delta conversion). v1 demos without the deviation flag "
            "are not loader-compatible — see handoff §1."
        )

    # State dim must match the env exactly. This catches the silent
    # 15-vs-21 trap from v1.
    file_state_dim = int(h5.attrs["state_dim"])
    if file_state_dim != env_shapes["state_dim"]:
        raise ValueError(
            f"state_dim mismatch: demo file has {file_state_dim}, live env produces "
            f"{env_shapes['state_dim']}. Demo file must be re-collected with the "
            "same env config that train_squint.py uses."
        )

    # RGB shape (H, W) — channels assumed 3
    file_h, file_w = int(h5.attrs["rgb_h"]), int(h5.attrs["rgb_w"])
    env_h, env_w = env_shapes["rgb_shape"][0], env_shapes["rgb_shape"][1]
    if (file_h, file_w) != (env_h, env_w):
        raise ValueError(
            f"rgb shape mismatch: demo {file_h}x{file_w}, env {env_h}x{env_w}. "
            "Set image_height/image_width or re-collect."
        )

    # Action bounds: ManiSkill envs report a [-1, 1] normalized single_action_space
    # universally (the controller internally scales by arm_delta_max / grip_delta_max).
    # The loader's job is to deliver normalized actions in [-1, 1]; the divisor for
    # that is the file's arm_delta_max / grip_delta_max.
    expected_low = np.array([-1.0] * 6, dtype=np.float32)
    expected_high = np.array([1.0] * 6, dtype=np.float32)
    if not (np.allclose(env_shapes["act_low"], expected_low, atol=1e-5) and
            np.allclose(env_shapes["act_high"], expected_high, atol=1e-5)):
        raise ValueError(
            "env action-space is not normalized [-1, 1] as expected:\n"
            f"  env low/high : {env_shapes['act_low']} / {env_shapes['act_high']}"
        )


def _iter_demos(h5: h5py.File):
    """Yield (group_name, group) for each demo_* group, in stored order."""
    keys = sorted(k for k in h5.keys() if k.startswith("demo_"))
    for k in keys:
        yield k, h5[k]


def load_demo_buffer(
    demo_file: str,
    env,
    device: torch.device,
    *,
    v_min: float = -20.0,
    v_max: float = 20.0,
    require_success: bool = True,
    verbose: bool = True,
) -> TensorDict:
    """Read demo_file (v2 schema), validate against env, return a TensorDict
    ready for `replay_buffer.extend(...)`.

    Each demo of length T contributes (T-1) transitions:
      (obs[t], action[t], reward[t], obs[t+1])  for t in 0..T-2.
    The final timestep is dropped because next_obs is undefined.

    Args:
        demo_file: path to the v2 h5 file
        env: the wrapped env (ManiSkillVectorEnv or its inner gym env). Used
            only for shape probing.
        device: target device for the returned TensorDict
        v_min, v_max: distributional critic support. Rewards outside this range
            would silently get clipped during projection, so the loader hard-
            fails if any demo reward escapes it.
        require_success: if True, skip demos whose `success` attr is False.
            v2 collector should only write successful demos, but this guards
            against schema drift.
        verbose: print a one-line summary on success

    Returns:
        TensorDict with batch_size = total number of transitions, keys:
            observations.rgb         : (N, H, W, 3) uint8
            observations.state       : (N, state_dim) float32
            next_observations.rgb    : (N, H, W, 3) uint8
            next_observations.state  : (N, state_dim) float32
            actions                  : (N, 6) float32
            rewards                  : (N,) float32
            dones                    : (N,) bool, all False
    """
    env_shapes = _get_env_shapes(env)

    with h5py.File(demo_file, "r") as h5:
        _validate_against_env(h5, env_shapes)

        H, W = int(h5.attrs["rgb_h"]), int(h5.attrs["rgb_w"])
        state_dim = int(h5.attrs["state_dim"])
        action_dim = int(h5.attrs["action_dim"])
        n_demos_total = int(h5.attrs["num_demos"])
        # Action conversion: native delta-pos demos use actions as-is; absolute-
        # joint-target demos need (delta = a[t] - a[t-1], then normalized by
        # [arm_max]*5 + [grip_max]). See handoff §1.
        needs_action_conversion = str(h5.attrs["control_mode"]) == "pd_joint_pos"
        arm_max = float(h5.attrs["arm_delta_max"])
        grip_max = float(h5.attrs["grip_delta_max"])
        action_divisor = np.array([arm_max] * 5 + [grip_max], dtype=np.float32)
        if needs_action_conversion and verbose:
            print(
                f"[demo_loader] control_mode='pd_joint_pos' → converting absolute "
                f"joint targets to normalized deltas (divisor={action_divisor.tolist()})"
            )

        # First pass: count transitions and filter by success.
        kept = []
        skipped_no_success = 0
        for name, g in _iter_demos(h5):
            if require_success and not bool(g.attrs.get("success", False)):
                skipped_no_success += 1
                continue
            T = int(g["actions"].shape[0])
            if T < 2:
                continue  # need at least one transition
            kept.append((name, g, T))

        if not kept:
            raise ValueError(
                f"no usable demos in {demo_file} "
                f"(total in file: {n_demos_total}, skipped for success: {skipped_no_success})"
            )

        n_trans = sum(T - 1 for _, _, T in kept)

        # Pre-allocate flat numpy buffers (cheap, single contiguous block per field).
        rgb_obs = np.empty((n_trans, H, W, 3), dtype=np.uint8)
        rgb_nxt = np.empty((n_trans, H, W, 3), dtype=np.uint8)
        st_obs = np.empty((n_trans, state_dim), dtype=np.float32)
        st_nxt = np.empty((n_trans, state_dim), dtype=np.float32)
        actions = np.empty((n_trans, action_dim), dtype=np.float32)
        rewards = np.empty((n_trans,), dtype=np.float32)

        # Second pass: fill.
        write = 0
        for name, g, T in kept:
            n = T - 1
            rgb = g["obs/rgb"][:]            # (T, H, W, 3) uint8
            st = g["obs/state"][:]           # (T, state_dim) float32
            act = g["actions"][:]            # (T, 6) float32
            rw = g["rewards"][:]             # (T,) float32

            if needs_action_conversion:
                # Per handoff §1: delta[t] = action[t] - action[t-1], with
                # action[-1] = QPOS_START. The synthesized target_qpos at
                # state[0, 3:9] IS the previous-step action (= QPOS_START at t=0),
                # so we can prepend it to recover the deltas without needing
                # the env's reset-qpos.
                qpos_start = st[0, 3:9].astype(np.float32)
                prev = np.vstack([qpos_start[None, :], act[:-1]])  # (T, 6)
                act = (act - prev) / action_divisor                # (T, 6) normalized delta

            # Per-transition reward range guard. Catches a demo that drifted
            # outside the distributional critic's [v_min, v_max] support.
            rw_min, rw_max = float(rw[:T - 1].min()), float(rw[:T - 1].max())
            if rw_min < v_min or rw_max > v_max:
                raise ValueError(
                    f"demo {name}: reward range [{rw_min:.3f}, {rw_max:.3f}] "
                    f"escapes critic support [{v_min}, {v_max}]"
                )

            # Per-transition action range guard. After conversion (if any),
            # actions should be normalized in roughly [-1, 1] to match the env.
            # Allow small overshoot tolerance for numerical noise / collector edge cases.
            a_abs_max = float(np.abs(act).max())
            if a_abs_max > 1.05:
                raise ValueError(
                    f"demo {name}: max |normalized action| = {a_abs_max:.5f} > 1.05 "
                    "after conversion — actions escape the env's normalized "
                    "[-1, 1] action space."
                )

            rgb_obs[write:write + n] = rgb[:T - 1]
            rgb_nxt[write:write + n] = rgb[1:T]
            st_obs[write:write + n] = st[:T - 1]
            st_nxt[write:write + n] = st[1:T]
            actions[write:write + n] = act[:T - 1]
            rewards[write:write + n] = rw[:T - 1]
            write += n

        assert write == n_trans, (write, n_trans)

    # Move to device. uint8 RGB stays as uint8 (matches online buffer).
    td = TensorDict(
        {
            "observations": TensorDict(
                {
                    "rgb": torch.as_tensor(rgb_obs, device=device),
                    "state": torch.as_tensor(st_obs, device=device),
                },
                batch_size=n_trans,
                device=device,
            ),
            "next_observations": TensorDict(
                {
                    "rgb": torch.as_tensor(rgb_nxt, device=device),
                    "state": torch.as_tensor(st_nxt, device=device),
                },
                batch_size=n_trans,
                device=device,
            ),
            "actions": torch.as_tensor(actions, device=device),
            "rewards": torch.as_tensor(rewards, device=device),
            # bootstrap_at_done='always' → dones must be False for the bootstrap
            # term to survive in the C51 target projection.
            "dones": torch.zeros(n_trans, dtype=torch.bool, device=device),
        },
        batch_size=n_trans,
        device=device,
    )

    if verbose:
        n_kept = len(kept)
        avg_T = n_trans / max(n_kept, 1)
        print(
            f"[demo_loader] {demo_file}: {n_kept} demos "
            f"(skipped {skipped_no_success} unsuccessful), "
            f"{n_trans} transitions, avg_T={avg_T:.1f}, "
            f"state_dim={state_dim}, reward range "
            f"[{float(rewards.min()):.3f}, {float(rewards.max()):.3f}]"
        )

    return td
