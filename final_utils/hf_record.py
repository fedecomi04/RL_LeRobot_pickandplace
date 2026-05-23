"""Episode recording → Hugging Face for the final_utils evals.

Reuses the deploy-run video + upload implementation already written in
infer_eval2_linux.py (`write_video`, `HFRunUploader`, the local `.hf_token`
reader) and wraps it in an `EpisodeRecorder` that plugs into run_pick_place /
run_split's `frame_sink` hook. Each recorded frame is a side-by-side composite:

    [ RAW camera frame | MASKED image actually fed to the policy ]

so the saved mp4 shows both what the camera saw and what the CNN saw. The
trajectory (qpos/target/action) and a metadata.json go alongside, and the run
folder is uploaded to a HF dataset repo (best-effort: no token / failure → the
local copy under out_dir is kept).

Recording is DEFAULT-ON in eval 1/2/3; the HF write token is read from the local
`.hf_token` file (or $HF_TOKEN). Frame capture is cheap (a resize + concat); the
mp4 encode + upload run AFTER the episode, so the control loop is never slowed.
"""
import datetime
import json
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np

# Reuse the existing implementation (single source of truth).
from deploy_utils.infer_eval2_linux import write_video, HFRunUploader, _read_token_file, _git_commit

GOAL_COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]


def _composite(raw_rgb, policy_rgb, raw_width):
    """RAW frame (left) | MASKED policy-input (right, nearest-upscaled to match
    height so its blocky pixels stay readable), with a thin white separator."""
    raw = np.asarray(raw_rgb)
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    if raw.shape[1] > raw_width:
        rh = int(round(raw_width * raw.shape[0] / raw.shape[1]))
        raw = cv2.resize(raw, (raw_width, rh), interpolation=cv2.INTER_AREA)
    h = raw.shape[0]
    pol = np.asarray(policy_rgb)
    if pol.ndim == 2:
        pol = cv2.cvtColor(pol, cv2.COLOR_GRAY2RGB)
    pw = max(1, int(round(h * pol.shape[1] / pol.shape[0])))
    pol = cv2.resize(pol, (pw, h), interpolation=cv2.INTER_NEAREST)
    sep = np.full((h, 4, 3), 255, np.uint8)
    return np.ascontiguousarray(np.concatenate([raw, sep, pol], axis=1), dtype=np.uint8)


class EpisodeRecorder:
    """Buffers a raw|masked composite mp4 + trajectory across a whole eval run,
    then writes a run folder and uploads it to Hugging Face.

    Use `recorder.add` as the `frame_sink` passed to run_split / run_pick_place;
    call `recorder.finish(success, goal=...)` once the run ends."""

    def __init__(self, *, enabled=True, upload=True, out_dir="deploy_runs",
                 eval_tier=1, hf_repo=None, hf_public=False,
                 raw_width=512, fps_cap=30, meta=None):
        self.enabled = enabled
        self.upload = upload
        self.out_dir = Path(out_dir)
        self.eval_tier = eval_tier
        self.hf_repo = hf_repo
        self.hf_public = hf_public
        self.raw_width = int(raw_width)
        self.min_dt = (1.0 / fps_cap) if fps_cap and fps_cap > 0 else 0.0
        self.meta = dict(meta or {})
        self.frames = []
        self.qpos, self.target, self.action = [], [], []
        self._t0 = self._tN = self._last_t = None

    def add(self, raw_rgb, policy_rgb, qpos, target_qpos, action_raw):
        if not self.enabled:
            return
        now = time.time()
        if self._last_t is not None and (now - self._last_t) < self.min_dt:
            return                                   # cap stored frame rate (memory)
        self._last_t = now
        if self._t0 is None:
            self._t0 = now
        self._tN = now
        self.frames.append(_composite(raw_rgb, policy_rgb, self.raw_width))
        self.qpos.append(np.asarray(qpos).copy())
        self.target.append(np.asarray(target_qpos).copy())
        self.action.append(np.asarray(action_raw).copy())

    def finish(self, success, *, goal=None):
        """Write video.mp4 (raw|masked) + trajectory.npz + metadata.json to a run
        dir and upload it. `goal` is an int or list of ints (logged in the name)."""
        if not self.enabled or not self.frames:
            return None
        ts = datetime.datetime.now()
        n = len(self.frames)
        dur = (self._tN - self._t0) if (self._t0 and self._tN and self._tN > self._t0) else None
        fps = (n - 1) / dur if (dur and n > 1) else 30.0
        rchar = "S" if success else "N"
        goals = [int(g) for g in (goal if isinstance(goal, (list, tuple)) else [goal])] if goal is not None else []
        gtag = ("_" + "-".join(GOAL_COLOR_NAMES[g] for g in goals)) if goals else ""
        run_name = f"{ts:%Y%m%d_%H%M%S}_eval{self.eval_tier}{gtag}_{rchar}"
        run_dir = self.out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        np.savez(run_dir / "trajectory.npz",
                 qpos=np.stack(self.qpos), target_qpos=np.stack(self.target),
                 action_raw=np.stack(self.action))
        video_ok = write_video(run_dir / "video.mp4", self.frames, fps)

        meta = {
            "date": ts.isoformat(timespec="seconds"),
            "eval": self.eval_tier,
            "goal_colors": goals,
            "goal_cube_names": [GOAL_COLOR_NAMES[g] for g in goals],
            "success": bool(success),
            "result": rchar,
            "n_steps": n,
            "video": "video.mp4 (left=raw camera | right=masked policy input)" if video_ok else None,
            "video_fps": int(round(fps)),
            "git_commit": _git_commit(),
            "host": socket.gethostname(),
        }
        meta.update(self.meta)
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"  [record] saved run → {run_dir}  (result={rchar}, {n} frames @ {fps:.1f} fps)")

        if self.upload:
            self._upload(run_dir, run_name)
        return run_dir

    def _upload(self, run_dir, run_name):
        token = os.environ.get("HF_TOKEN") or _read_token_file()
        if not token:
            print("  [hf] no token (.hf_token / $HF_TOKEN) — run kept local only.")
            return
        try:
            from huggingface_hub import HfApi
            repo = self.hf_repo or f"{HfApi(token=token).whoami()['name']}/squint-deploy-runs"
            up = HFRunUploader(token, repo, private=not self.hf_public)
            remote = up.upload_run(run_dir, run_name)
            print(f"  [hf] uploaded → https://huggingface.co/datasets/{repo}/tree/main/{remote}")
        except Exception as e:
            print(f"  [hf] upload failed ({e}); local copy kept at {run_dir}")
