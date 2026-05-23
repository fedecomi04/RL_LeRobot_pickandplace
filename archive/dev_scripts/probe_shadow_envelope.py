"""Probe how many parallel envs SAPIEN can fit at shadows=True / 1 directional
light / training-resolution sensor.

Each value of num_envs is tested in a FRESH process via subprocess.Popen — the
SAPIEN parallel renderer pre-sizes its shadow caster pool on the first scene
build of a process, so probing sequentially in one process gives misleading
"works" / "fails" splits.

Prints the largest num_envs that succeeds, in JSON to --out.

Usage (on the Brev VM):
    python scripts/probe_shadow_envelope.py --out /tmp/shadow_envelope.json
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Test progression, biggest first (so we stop at first success).
DEFAULT_PROBE_SIZES = [1024, 768, 512, 384, 256, 192, 128, 96, 64, 32]


WORKER_SOURCE = textwrap.dedent(
    """
    import os, sys, gymnasium as gym, torch
    sys.path.insert(0, os.path.expanduser('~/squint'))
    import envs  # noqa
    n_envs = int(sys.argv[1])
    image_h = int(sys.argv[2])
    image_w = int(sys.argv[3])
    try:
        env = gym.make(
            'SO101PlaceCube-v1', num_envs=n_envs,
            obs_mode='rgb', render_mode='rgb_array', sim_backend='gpu',
            domain_randomization=True,
            domain_randomization_config={
                'shadows': True,
                'num_directional_lights': 1,
                'num_point_lights': 0,
            },
            control_mode='pd_joint_target_delta_pos',
            sensor_configs=dict(width=image_w, height=image_h),
            n_distractors=1,
            use_real_bowl=True,
            split_only_reward=True,
            split_target_gap=0.025,
        )
        obs, _ = env.reset(seed=0)
        # Force a sensor render so the parallel renderer actually allocates.
        rgb_shape = tuple(obs['sensor_data']['hand_camera']['rgb'].shape) if 'sensor_data' in obs else tuple(obs['rgb'].shape) if 'rgb' in obs else 'n/a'
        # Free
        env.close()
        print('PROBE_OK n_envs=' + str(n_envs) + ' rgb_shape=' + str(rgb_shape))
        sys.exit(0)
    except RuntimeError as e:
        print('PROBE_FAIL n_envs=' + str(n_envs) + ' err=' + str(e).splitlines()[0][:200])
        sys.exit(2)
    except Exception as e:
        print('PROBE_ERR n_envs=' + str(n_envs) + ' err=' + type(e).__name__ + ': ' + str(e)[:200])
        sys.exit(3)
    """
).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/shadow_envelope.json")
    ap.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_PROBE_SIZES),
        help="comma-separated num_envs values to try, biggest first",
    )
    ap.add_argument("--image_height", type=int, default=80)
    ap.add_argument("--image_width", type=int, default=144)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    worker_path = Path("/tmp/_shadow_probe_worker.py")
    worker_path.write_text(WORKER_SOURCE)

    results = []
    success_size = None
    for n in sizes:
        print(f"probing num_envs={n} ...", flush=True)
        proc = subprocess.run(
            [
                sys.executable, str(worker_path), str(n),
                str(args.image_height), str(args.image_width),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        out_line = (proc.stdout or proc.stderr or "").strip().splitlines()
        first = out_line[-1] if out_line else "<no output>"
        rec = {
            "num_envs": n,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "summary": first,
        }
        results.append(rec)
        print(f"  -> {first}", flush=True)
        if proc.returncode == 0 and success_size is None:
            success_size = n
            break  # found the largest; no need to keep probing smaller

    out = {
        "image_h": args.image_height,
        "image_w": args.image_width,
        "shadows": True,
        "num_directional_lights": 1,
        "sizes_tried_largest_first": sizes,
        "largest_fitting_num_envs": success_size,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}", flush=True)
    if success_size is None:
        print("NO num_envs succeeded — shadow fine-tune infeasible at these sizes",
              flush=True)
        sys.exit(1)
    print(f"LARGEST FITTING num_envs = {success_size}", flush=True)


if __name__ == "__main__":
    main()
