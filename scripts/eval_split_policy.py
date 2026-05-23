"""Headless multi-episode evaluator for split-task policies.

Runs N parallel rollouts (one env per episode), captures per-step trajectories,
and computes the four metrics the user actually cares about:

    final_gap            (m)   surface gap between cubes at episode end / first success
    max_cube_disp        (m)   max over cubes of ||xy_final - xy_spawn||
    max_gripper_drift    (rad) max over t of |qpos_gripper[t] - qpos_gripper[0]|
    mean_action_jitter   (-)   mean over t of ||a_t - a_{t-1}||  (post-tanh action)
    tcp_max_speed        (m/s) max over t of ||tcp[t] - tcp[t-1]|| * control_hz

A composite "success" predicate (--gap_lo/--gap_hi/--max_disp_max/etc.) is also
reported. Default thresholds reflect the user's spec:

    final_gap         in [0.020, 0.035] m
    max_cube_disp     < 0.05 m
    max_gripper_drift < 0.05 rad
    mean_jitter       < 0.10  (post-tanh, 6-dim L2 per step)

Writes a JSON summary to --out. Optionally writes mp4s of the first --n_videos
envs to --video_dir for visual sanity-check.

Usage (on the Brev VM):
    python scripts/eval_split_policy.py \
        --checkpoint runs/split_2cube_hover_80x144/ckpt.pt \
        --n_episodes 100 --n_distractors 1 \
        --out /tmp/eval_existing.json --video_dir /tmp/eval_vids
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import cv2

SIM_CONTROL_HZ = 10  # must match training's control_freq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from train_squint import CNNEncoder, Actor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101PlaceCube-v1")
    ap.add_argument("--n_episodes", type=int, default=100,
                    help="number of parallel envs / episodes to evaluate")
    ap.add_argument("--n_distractors", type=int, default=1,
                    help="cubes other than goal cube; 1 = the eval2 2-cube scene")
    ap.add_argument("--max_steps", type=int, default=100,
                    help="control steps per episode (training default is 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain_randomization", action="store_true", default=True,
                    help="match training distribution (default True)")
    ap.add_argument("--no_dr", action="store_false", dest="domain_randomization")
    ap.add_argument("--shadows", action="store_true", default=False,
                    help="enable DR directional-light shadows; off by default to "
                         "avoid the SAPIEN 'too many shadow-casting lights' OOM "
                         "when n_episodes × render_size is large (matches "
                         "scripts/brev_run_split.sh SHADOWS=false default)")
    ap.add_argument("--image_height", type=int, default=80)
    ap.add_argument("--image_width", type=int, default=144)
    ap.add_argument("--out", default="/tmp/eval_split.json")
    ap.add_argument("--video_dir", default="",
                    help="if non-empty, save first --n_videos as mp4s here")
    ap.add_argument("--n_videos", type=int, default=5)
    ap.add_argument("--render_size", type=int, default=256)
    # success thresholds (defaults reflect the user's spec)
    ap.add_argument("--gap_lo", type=float, default=0.020)
    ap.add_argument("--gap_hi", type=float, default=0.035)
    ap.add_argument("--max_disp_max", type=float, default=0.05)
    ap.add_argument("--gripper_drift_max", type=float, default=0.05)
    ap.add_argument("--mean_jitter_max", type=float, default=0.10)
    return ap.parse_args()


def main():
    args = parse_args()
    save_videos = bool(args.video_dir)
    if save_videos:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=args.domain_randomization,
        domain_randomization_config={"shadows": args.shadows},
        control_mode="pd_joint_target_delta_pos",
        sensor_configs=dict(width=640, height=480),
        human_render_camera_configs=dict(
            shader_pack="default", width=args.render_size, height=args.render_size
        ),
        n_distractors=args.n_distractors,
        use_real_bowl=True,
        # Switch the env into split-task mode so evaluate() exposes
        # info["min_gap"] and info["success"] reflects "cubes separated +
        # static" rather than the default "placed in bowl" criterion.
        # Matches what the existing ckpt was trained against.
        split_only_reward=True,
        split_target_gap=0.025,
        split_hover_after_separate=True,
    )
    env = gym.make(args.env_id, num_envs=args.n_episodes, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    base = env.unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    obs_space = env.unwrapped.single_observation_space
    n_state = obs_space["state"].shape[0]
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(
        n_obs=(args.image_height, args.image_width, 3), device=device
    ).to(device)
    actor = Actor(
        env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()
    actor.eval()
    print(f"loaded {args.checkpoint} @ step {ckpt.get('global_step')}", flush=True)
    print(f"running {args.n_episodes} episodes x {args.max_steps} steps "
          f"(DR={args.domain_randomization}, n_distractors={args.n_distractors})",
          flush=True)

    obs, _ = env.reset(seed=args.seed)
    E = args.n_episodes
    T = args.max_steps
    n_cubes = args.n_distractors + 1

    # Pre-allocate per-step buffers. cube_xyz stores all 3 dims so we can
    # recompute min_gap defensively from cube centers + item_half_sizes if
    # the env doesn't propagate info["min_gap"].
    cube_xyz = torch.zeros((T + 1, E, n_cubes, 3), device=device)
    tcp_pos = torch.zeros((T + 1, E, 3), device=device)
    grip_q = torch.zeros((T + 1, E), device=device)
    actions = torch.zeros((T, E, n_act), device=device)
    min_gap = torch.zeros((T, E), device=device)
    item_half_sizes = base.item_half_sizes.detach().clone().to(device)  # (E,)
    success_step = torch.full((E,), -1, dtype=torch.long, device=device)
    done_step = torch.full((E,), -1, dtype=torch.long, device=device)
    scene_frames = [] if save_videos else None  # list of (H, W, 3) frames, post-cat

    def read_cube_xyz():
        cubes = [base.item.pose.p]
        for d in base.distractors:
            cubes.append(d.pose.p)
        return torch.stack(cubes, dim=1).detach().clone()  # (E, C, 3)

    def compute_min_gap_from_xyz(xyz_step):
        """Surface gap = min pairwise center distance − 2·half_size.
        Matches envs/place.py:1465-1470."""
        diff = xyz_step[:, :, None, :] - xyz_step[:, None, :, :]  # (E, C, C, 3)
        d = torch.linalg.norm(diff, dim=-1)  # (E, C, C)
        # exclude self-pairs by setting diagonal to +inf
        big = torch.full_like(d, float("inf"))
        d = torch.where(torch.eye(n_cubes, device=device, dtype=torch.bool)[None],
                        big, d)
        min_center_dist = d.amin(dim=(1, 2))  # (E,)
        return min_center_dist - 2.0 * item_half_sizes

    def read_tcp():
        return base.agent.tcp_pose.p.detach().clone()  # (E, 3)

    def read_grip():
        return base.agent.robot.get_qpos()[:, -1].detach().clone()  # (E,)

    # Initial state (step 0).
    cube_xyz[0] = read_cube_xyz()
    tcp_pos[0] = read_tcp()
    grip_q[0] = read_grip()

    t_wall0 = time.perf_counter()
    for t in range(T):
        rgb_now = obs["rgb"]
        state_now = obs["state"]
        if not torch.is_tensor(rgb_now):
            rgb_now = torch.from_numpy(rgb_now)
        if not torch.is_tensor(state_now):
            state_now = torch.from_numpy(state_now)
        rgb_t = rgb_now.permute(0, 3, 1, 2).float()
        rgb_small = F.interpolate(
            rgb_t, size=(args.image_height, args.image_width), mode="area"
        ).permute(0, 2, 3, 1).to(torch.uint8)

        with torch.no_grad():
            feats = encoder(rgb_small.to(device))
            mean = actor.forward(feats, state_now.float().to(device))
            action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)
        actions[t] = action.detach()

        if save_videos and t % 1 == 0:  # every step
            scene = env.render()
            if torch.is_tensor(scene):
                scene = scene.detach().cpu().numpy()
            scene_frames.append(np.asarray(scene)[: args.n_videos].astype(np.uint8))

        obs, rew, term, trunc, info = env.step(
            action.detach().cpu().numpy().astype(np.float32)
        )

        # Post-step cube state — used to compute min_gap defensively even if
        # the env didn't expose info["min_gap"] (only set in split-only mode,
        # envs/place.py:1454). Also lets us recover the post-step state for
        # final_gap measurement.
        cube_xyz_post = read_cube_xyz()
        if "min_gap" in info:
            mg = info["min_gap"]
            min_gap[t] = (mg.detach().clone() if torch.is_tensor(mg)
                          else torch.as_tensor(mg, device=device))
        else:
            min_gap[t] = compute_min_gap_from_xyz(cube_xyz_post)
        if "success" in info:
            succ = info["success"]
            succ_t = succ.detach() if torch.is_tensor(succ) else torch.as_tensor(succ, device=device)
            newly_succ = (succ_t > 0.5) & (success_step < 0)
            success_step = torch.where(newly_succ, torch.full_like(success_step, t), success_step)

        # Detect first done (term or trunc).
        term_t = torch.as_tensor(term, device=device).bool() if not torch.is_tensor(term) else term.bool()
        trunc_t = torch.as_tensor(trunc, device=device).bool() if not torch.is_tensor(trunc) else trunc.bool()
        any_done = term_t | trunc_t
        newly_done = any_done & (done_step < 0)
        done_step = torch.where(newly_done, torch.full_like(done_step, t), done_step)

        # Snapshot state for the NEXT iteration. With auto-reset, the next
        # iter's pre-step state would be reset-state for any env that just
        # terminated. We only ever read pre-step buffers up to done_step.
        cube_xyz[t + 1] = cube_xyz_post
        tcp_pos[t + 1] = read_tcp()
        grip_q[t + 1] = read_grip()

    wall = time.perf_counter() - t_wall0
    sim_time = T / SIM_CONTROL_HZ
    print(f"rollout done in {wall:.1f} s wall ({sim_time:.1f} s sim, "
          f"{sim_time / wall:.2f}x), now computing metrics", flush=True)

    # Per-env end_step: success_step if it triggered, else done_step, else T-1.
    end_step = torch.where(success_step >= 0, success_step, done_step)
    end_step = torch.where(end_step >= 0, end_step, torch.full_like(end_step, T - 1))

    end_step_cpu = end_step.cpu().numpy()
    success_mask = (success_step >= 0).cpu().numpy()

    final_gap = np.zeros(E)
    max_disp = np.zeros(E)
    max_grip_drift = np.zeros(E)
    mean_jitter = np.zeros(E)
    tcp_max_speed = np.zeros(E)

    cube_xy_np = cube_xyz[..., :2].cpu().numpy()  # (T+1, E, C, 2)
    tcp_pos_np = tcp_pos.cpu().numpy()      # (T+1, E, 3)
    grip_q_np = grip_q.cpu().numpy()        # (T+1, E)
    actions_np = actions.cpu().numpy()      # (T, E, n_act)
    min_gap_np = min_gap.cpu().numpy()      # (T, E)

    for e in range(E):
        k = max(1, int(end_step_cpu[e]) + 1)  # use steps 0..k-1 in buffers
        final_gap[e] = float(min_gap_np[k - 1, e])
        # displacements from spawn (xy only)
        disp = np.linalg.norm(
            cube_xy_np[: k + 1, e] - cube_xy_np[0:1, e], axis=-1
        )  # (k+1, C)
        max_disp[e] = float(disp.max())
        # gripper drift
        max_grip_drift[e] = float(np.abs(grip_q_np[: k + 1, e] - grip_q_np[0, e]).max())
        # action jitter (post-tanh)
        if k >= 2:
            da = actions_np[1:k, e] - actions_np[: k - 1, e]
            mean_jitter[e] = float(np.linalg.norm(da, axis=-1).mean())
        else:
            mean_jitter[e] = 0.0
        # TCP speed (m/s)
        if k >= 2:
            dtcp = np.linalg.norm(
                tcp_pos_np[1: k + 1, e] - tcp_pos_np[: k, e], axis=-1
            )
            tcp_max_speed[e] = float(dtcp.max() * SIM_CONTROL_HZ)
        else:
            tcp_max_speed[e] = 0.0

    # User's composite predicate.
    pred_gap = (final_gap >= args.gap_lo) & (final_gap <= args.gap_hi)
    pred_disp = max_disp < args.max_disp_max
    pred_grip = max_grip_drift < args.gripper_drift_max
    pred_jit = mean_jitter < args.mean_jitter_max
    pred_all = pred_gap & pred_disp & pred_grip & pred_jit

    def pct(mask):
        return float(np.asarray(mask).mean())

    def stats(x):
        return dict(
            mean=float(np.mean(x)),
            median=float(np.median(x)),
            p10=float(np.percentile(x, 10)),
            p90=float(np.percentile(x, 90)),
            min=float(np.min(x)),
            max=float(np.max(x)),
        )

    summary = dict(
        checkpoint=args.checkpoint,
        ckpt_global_step=int(ckpt.get("global_step", -1) or -1),
        n_episodes=E,
        max_steps=T,
        n_distractors=args.n_distractors,
        domain_randomization=args.domain_randomization,
        thresholds=dict(
            gap_lo=args.gap_lo,
            gap_hi=args.gap_hi,
            max_disp_max=args.max_disp_max,
            gripper_drift_max=args.gripper_drift_max,
            mean_jitter_max=args.mean_jitter_max,
        ),
        env_success_rate=pct(success_mask),
        composite_pass_rate=pct(pred_all),
        per_predicate_pass_rates=dict(
            gap_in_band=pct(pred_gap),
            cube_disp_ok=pct(pred_disp),
            gripper_quiet=pct(pred_grip),
            action_smooth=pct(pred_jit),
        ),
        metrics=dict(
            final_gap_m=stats(final_gap),
            max_cube_disp_m=stats(max_disp),
            max_gripper_drift_rad=stats(max_grip_drift),
            mean_action_jitter=stats(mean_jitter),
            tcp_max_speed_mps=stats(tcp_max_speed),
        ),
        wall_seconds=wall,
    )

    print(json.dumps(summary, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args.out}", flush=True)

    if save_videos and scene_frames:
        # scene_frames is list (len T) of arrays (n_videos, H, W, 3).
        # OpenCV's "mp4v" fourcc writes MPEG-4 Part 2, which macOS QuickTime
        # refuses to open. Prefer ffmpeg → libx264 (H.264 / AVC) for QuickTime
        # + browser compatibility; fall back to cv2.mp4v if ffmpeg is absent.
        n_v = scene_frames[0].shape[0]
        H, W = scene_frames[0].shape[1:3]
        has_ffmpeg = shutil.which("ffmpeg") is not None
        for vi in range(n_v):
            out_mp4 = Path(args.video_dir) / f"ep{vi:02d}.mp4"
            if has_ffmpeg:
                proc = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{W}x{H}", "-r", str(SIM_CONTROL_HZ),
                        "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(out_mp4),
                    ],
                    stdin=subprocess.PIPE,
                )
                for fi, frames in enumerate(scene_frames):
                    bgr = cv2.cvtColor(frames[vi], cv2.COLOR_RGB2BGR)
                    cv2.rectangle(bgr, (0, 0), (W, 22), (0, 0, 0), -1)
                    lbl = f"ep{vi} t{fi:3d} gap={float(min_gap_np[fi, vi]) if fi < T else 0:.3f}m"
                    cv2.putText(bgr, lbl, (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
                    proc.stdin.write(bgr.tobytes())
                proc.stdin.close()
                proc.wait()
            else:
                # Fallback: mpeg4 — playable in VLC / mpv but not QuickTime.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(out_mp4), fourcc, SIM_CONTROL_HZ, (W, H))
                for fi, frames in enumerate(scene_frames):
                    bgr = cv2.cvtColor(frames[vi], cv2.COLOR_RGB2BGR)
                    cv2.rectangle(bgr, (0, 0), (W, 22), (0, 0, 0), -1)
                    lbl = f"ep{vi} t{fi:3d} gap={float(min_gap_np[fi, vi]) if fi < T else 0:.3f}m"
                    cv2.putText(bgr, lbl, (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
                    vw.write(bgr)
                vw.release()
            print(f"saved {out_mp4}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
