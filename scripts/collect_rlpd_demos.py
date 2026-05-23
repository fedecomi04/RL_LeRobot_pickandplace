#!/usr/bin/env python3
"""
Collect 50 RLPD demos using the working V3 recipe (gym.make only, no
ManiSkillVectorEnv wrapper, pd_joint_pos absolute targets). num_envs=8
batched for speed. Output matches the v2 HDF5 schema, with the same
collector_deviation_* attrs already validated in the 2-demo pipeline test.

Sanity videos: 1 per color (6 total), saved alongside the h5.
"""
import sys, os, math, json, time, datetime, subprocess, argparse
import numpy as np, torch, gymnasium as gym, h5py, cv2
sys.path.insert(0, '/home/shadeform/squint')
import envs
from deploy_utils.so101_fk import tcp_pos, nudge_arm_joints, finger_positions

# CLI args (D4-A from plan 2026-05-21_0545): T_* and run sizing are flags so
# we can iterate without editing the file. Defaults = the "medium" recipe
# (~270 control steps = ~9s @ 30Hz real-robot replay, ~3x faster than the
# original 785-step recipe).
def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="/tmp/rlpd_50demos_v4_medium")
    ap.add_argument("--num_demos", type=int, default=50)
    ap.add_argument("--max_batches", type=int, default=35,
                    help="upper bound on collection batches; ~30%% IK grasp rate → ~21 batches for 50 demos, 35 = margin")
    ap.add_argument("--t_home", type=int, default=10)
    ap.add_argument("--t_move", type=int, default=55)
    ap.add_argument("--t_desc", type=int, default=30)
    ap.add_argument("--t_grip", type=int, default=25)
    ap.add_argument("--t_grasp_hold", type=int, default=12)
    ap.add_argument("--t_lift", type=int, default=35)
    ap.add_argument("--t_trans", type=int, default=55)
    ap.add_argument("--t_hover", type=int, default=10)
    ap.add_argument("--t_rel", type=int, default=18)
    ap.add_argument("--bound_arm", type=float, default=0.05,
                    help="arm-joint delta bound (rad/step) for the pre-write check")
    ap.add_argument("--bound_grip", type=float, default=0.20,
                    help="gripper delta bound (rad/step) for the pre-write check")
    ap.add_argument("--bound_max_normalized", type=float, default=1.0,
                    help="reject demo if peak |action[t]-action[t-1]| / bound > this. "
                         "Strict (1.0) is tighter than the loader's 1.05 cap (D2-A in plan).")
    return ap.parse_args()

ARGS = _parse_args()

OUT_DIR = ARGS.out_dir
OUT_H5  = f'{OUT_DIR}/demos.h5'
OUT_META= f'{OUT_DIR}/meta.json'
NUM_DEMOS_TARGET = ARGS.num_demos
NUM_COLORS  = 6
BATCH_SIZE  = 8
MAX_BATCHES = ARGS.max_batches
PER_COLOR_MAX = (NUM_DEMOS_TARGET + NUM_COLORS - 1) // NUM_COLORS  # 9
IMG_H, IMG_W = 80, 144
N_ARM = 5

GRIPPER_OPEN_FULL = np.float32(120 * math.pi / 180)
GRIPPER_OPEN_DESC = np.float32(60  * math.pi / 180)
GRIPPER_CLOSED    = np.float32(5   * math.pi / 180)
GRIPPER_CLOSED_F  = float(GRIPPER_CLOSED)

QPOS_START = np.array([
    0.0, -80.791*math.pi/180, 36.747*math.pi/180,
    86.901*math.pi/180, -82.154*math.pi/180, GRIPPER_OPEN_FULL,
], dtype=np.float64)

T_HOME, T_MOVE, T_DESC = ARGS.t_home, ARGS.t_move, ARGS.t_desc
T_GRIP, T_GRASP_HOLD   = ARGS.t_grip, ARGS.t_grasp_hold
T_LIFT, T_TRANS, T_HOVER, T_REL = ARGS.t_lift, ARGS.t_trans, ARGS.t_hover, ARGS.t_rel

# Strict pre-write bound check divisor (D2-A). Symmetric with demo_loader.py's
# divisor; if peak |delta/divisor| > ARGS.bound_max_normalized (1.0) the demo
# is dropped before the HDF5 write, with a one-line reason printed.
BOUND_DIVISOR = np.array([ARGS.bound_arm]*5 + [ARGS.bound_grip], dtype=np.float32)

COLOR_NAMES = ['red','blue','green','yellow','purple','orange']

# ─── IK + face-to-face wrist alignment ───────────────────────────────────────
def solve_ik(q0, target_tcp, n_outer=200, step_frac=0.05, tol=0.002):
    q = np.array(q0, dtype=np.float64).copy()
    for _ in range(n_outer):
        cur = tcp_pos(q); e = target_tcp - cur; err = np.linalg.norm(e)
        if err < tol: return q
        step = e * min(1.0, step_frac/err)
        q = q + nudge_arm_joints(q, step, max_joint_step=0.30, iters=8)
    return q

def closed_jaw_xy_angle(q):
    qc = q.copy(); qc[5] = GRIPPER_CLOSED_F
    f1, f2 = finger_positions(qc)
    v = (f2 - f1)[:2]
    if np.linalg.norm(v) < 1e-5: return 0.0
    return math.atan2(v[1], v[0])

def cube_theta_from_quat(q_wxyz):
    return 2 * math.atan2(q_wxyz[3], q_wxyz[0])

def best_face_delta(cur, theta):
    best, ba = 0.0, math.inf
    for k in range(4):
        d = (theta + k*math.pi/2) - cur
        d = ((d + math.pi) % (2*math.pi)) - math.pi
        if abs(d) < ba: best, ba = d, abs(d)
    return best

def align_wrist_roll(q, theta, lo=-2.74, hi=2.84):
    delta = best_face_delta(closed_jaw_xy_angle(q), theta)
    qn = q.copy()
    qn[4] = max(lo+0.05, min(hi-0.05, qn[4] + delta))
    return qn

def plan_env(cube_pos, cube_quat, bowl_pos, cube_half, bowl_hz, q_init):
    theta = cube_theta_from_quat(cube_quat)
    cube_top = cube_pos[2] + cube_half
    bowl_rim = bowl_pos[2] + 2 * bowl_hz
    tcp_pre   = np.array([cube_pos[0], cube_pos[1], cube_top + 0.10])
    tcp_grasp = np.array([cube_pos[0], cube_pos[1], cube_pos[2] - 0.003])
    tcp_bowl  = np.array([bowl_pos[0], bowl_pos[1], bowl_rim + 0.04])
    q_pre = solve_ik(q_init, tcp_pre)
    q0g = q_pre.copy(); q0g[5] = GRIPPER_CLOSED
    q_grasp = solve_ik(q0g, tcp_grasp)
    q_grasp = align_wrist_roll(q_grasp, theta)
    q_grasp = solve_ik(q_grasp, tcp_grasp)
    q_grasp = align_wrist_roll(q_grasp, theta)
    q_lift = q_grasp.copy(); q_lift[1] -= 0.55; q_lift[2] += 0.20
    q_bowl = solve_ik(q_lift, tcp_bowl)
    return dict(q_pre=q_pre, q_grasp=q_grasp, q_lift=q_lift, q_bowl=q_bowl)

def cos_t(i, n): return (1 - math.cos(math.pi * i / max(n-1, 1))) / 2

def lerp_seq(q0, q1, n, hold_n):
    out = [((1-cos_t(i,n))*q0 + cos_t(i,n)*q1) for i in range(n)]
    out += [q1.copy() for _ in range(hold_n)]
    return out

def build_abs_sequence(plan, q_init):
    qi = q_init.astype(np.float32);    qi[5] = GRIPPER_OPEN_FULL
    qpre  = plan['q_pre'].astype(np.float32);   qpre[5]  = GRIPPER_OPEN_FULL
    qg_op = plan['q_grasp'].astype(np.float32); qg_op[5] = GRIPPER_OPEN_DESC
    qg_cl = plan['q_grasp'].astype(np.float32); qg_cl[5] = GRIPPER_CLOSED
    ql    = plan['q_lift'].astype(np.float32);  ql[5]    = GRIPPER_CLOSED
    qb    = plan['q_bowl'].astype(np.float32);  qb[5]    = GRIPPER_CLOSED
    qr    = plan['q_bowl'].astype(np.float32);  qr[5]    = GRIPPER_OPEN_FULL
    seq  = lerp_seq(qi,     qpre,  T_MOVE + T_HOME, T_HOME)
    seq += lerp_seq(qpre,   qg_op, T_DESC, 25)
    seq += lerp_seq(qg_op,  qg_cl, T_GRIP, T_GRASP_HOLD)
    seq += lerp_seq(qg_cl,  ql,    T_LIFT, T_HOME)
    seq += lerp_seq(ql,     qb,    T_TRANS, T_HOVER)
    seq += lerp_seq(qb,     qr,    T_REL,  T_HOME)
    return np.stack(seq, axis=0).astype(np.float32)

# ─── Env: gym.make only (no ManiSkillVectorEnv) ─────────────────────────────
print(f"Building env (V3 recipe, num_envs={BATCH_SIZE}, no MSVecEnv)...")
env = gym.make(
    'SO101PlaceCube-v1',
    num_envs=BATCH_SIZE,
    obs_mode='rgb',
    render_mode='all',
    sim_backend='gpu',
    sensor_configs=dict(width=640, height=360),
    domain_randomization=False,
    n_distractors=0,
    use_real_bowl=True,
    control_mode='pd_joint_pos',
    sim_freq=100, control_freq=10,
    pick_only_reward=False, split_only_reward=False, action_smooth_coef=0.0,
)
ue = env.unwrapped
dev = torch.device('cuda:0')

# Probe obs to get state layout & sizes
obs0, _ = env.reset(seed=0)
# obs0 is a dict from the base env (not flattened). We'll extract rgb + build state manually.

def make_state(obs_d):
    """Concat to alphabetical-key order to mirror FlattenRGBDObservationWrapper.
       Returns (B, 15) float32: bowl_xyz(3) + goal_color(6) + noisy_qpos(6)."""
    parts = []
    a = obs_d['agent']
    for k in sorted(a.keys()):
        v = a[k]
        if torch.is_tensor(v):
            v = v.reshape(v.shape[0], -1) if v.ndim > 1 else v.unsqueeze(-1)
            parts.append(v)
    return torch.cat(parts, dim=-1).cpu().numpy().astype(np.float32)

def make_rgb_downsampled(obs_d):
    """Take obs sensor rgb (B,360,640,3) → (B, IMG_H, IMG_W, 3) uint8."""
    rgb_raw = obs_d['sensor_data']['base_camera']['rgb']
    if torch.is_tensor(rgb_raw): rgb_raw = rgb_raw.cpu().numpy()
    rgb_raw = np.asarray(rgb_raw)
    B = rgb_raw.shape[0]
    out = np.zeros((B, IMG_H, IMG_W, 3), dtype=np.uint8)
    for i in range(B):
        out[i] = cv2.resize(rgb_raw[i], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return out

state15_dim = make_state(obs0).shape[-1]
print(f"  raw state_dim = {state15_dim}  (will augment to 21 with target_qpos at save time)")

# ─── Main loop ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
demos = []
color_counts = [0]*NUM_COLORS
hires_per_batch = []   # (batch_i, frames_array) for selected demos
saved_video_for_color = set()

def pick_colors():
    sc = sorted(range(NUM_COLORS), key=lambda c: color_counts[c])
    return [sc[i % NUM_COLORS] for i in range(BATCH_SIZE)]

t0 = time.time()
for batch_i in range(MAX_BATCHES):
    if len(demos) >= NUM_DEMOS_TARGET: break
    seed = batch_i * 1000 + 7
    obs, _ = env.reset(seed=seed)
    qstart_act = QPOS_START[np.newaxis].astype(np.float32).repeat(BATCH_SIZE, axis=0)

    # Override colors per env
    target_colors = pick_colors()
    color_tensor = torch.tensor(target_colors, device=dev, dtype=torch.long)
    ue.goal_color_idx[:] = color_tensor
    ue._set_actor_palette_color(ue.item,
                                 torch.arange(BATCH_SIZE, device=dev),
                                 color_tensor)

    # Stabilize at QPOS_START
    for _ in range(60):
        obs, _, _, _, _ = env.step(qstart_act)

    # Re-apply colors after stabilize (in case anything reset them)
    ue.goal_color_idx[:] = color_tensor
    ue._set_actor_palette_color(ue.item,
                                 torch.arange(BATCH_SIZE, device=dev),
                                 color_tensor)
    obs, _, _, _, _ = env.step(qstart_act)   # one step so obs reflects new colors

    cube_p = ue.item.pose.p.cpu().numpy()
    cube_q = ue.item.pose.q.cpu().numpy()
    bowl_p = ue.bin.pose.p.cpu().numpy()
    cube_half = ue.item_half_sizes[0].cpu().item()
    bowl_hz = getattr(ue, 'bowl_half_z', 0.0265)
    q_init = ue.agent.robot.get_qpos()[:, :6].cpu().numpy()

    # Plan IK per env
    abs_seqs = []
    for i in range(BATCH_SIZE):
        plan = plan_env(cube_p[i], cube_q[i], bowl_p[i], cube_half, bowl_hz,
                         q_init[i].astype(np.float64))
        abs_seqs.append(build_abs_sequence(plan, q_init[i]))
    actions_batch = np.stack(abs_seqs, axis=1)   # (T, B, 6) absolute targets
    T = actions_batch.shape[0]

    # Decide whether to record hi-res frames for sanity videos
    # (any color in this batch we haven't yet saved a video for)
    record_hires = any(c not in saved_video_for_color for c in target_colors)

    rgb_buf, st_buf, act_buf, rew_buf, done_buf = [], [], [], [], []
    hires_buf = []
    for t in range(T):
        rgb_buf.append(make_rgb_downsampled(obs))
        st_buf.append(make_state(obs))
        if record_hires:
            r = ue.render()
            if torch.is_tensor(r): r = r.cpu().numpy()
            hires_buf.append(np.asarray(r))  # (B, H, W, 3)
        a = actions_batch[t]
        act_buf.append(a.copy())
        obs, rew, term, trunc, info = env.step(a)
        rew_buf.append(rew.cpu().numpy() if torch.is_tensor(rew) else np.asarray(rew))
        done_buf.append((term | trunc).cpu().numpy() if torch.is_tensor(term)
                          else np.asarray(term | trunc))

    rgb_arr   = np.stack(rgb_buf,  axis=1)        # (B, T, H, W, 3)
    st15_arr  = np.stack(st_buf,   axis=1)        # (B, T, 15)
    act_arr   = np.stack(act_buf,  axis=1)        # (B, T, 6) absolute
    rew_arr   = np.stack(rew_buf,  axis=1)
    done_arr  = np.stack(done_buf, axis=1)

    final_cube_xy = ue.item.pose.p[:, :2].cpu().numpy()
    final_cube_z  = ue.item.pose.p[:, 2].cpu().numpy()
    dist = np.linalg.norm(final_cube_xy - bowl_p[:, :2], axis=1)

    n_added = 0
    for i in range(BATCH_SIZE):
        in_bowl = (dist[i] < 0.05) and (final_cube_z[i] > bowl_p[i, 2])
        if not in_bowl: continue
        color = target_colors[i]
        if color_counts[color] >= PER_COLOR_MAX: continue

        # Augment state 15→21: insert target_qpos (= previous action) at index 3.
        tq = np.zeros((T, 6), dtype=np.float32)
        tq[0]  = q_init[i].astype(np.float32)
        tq[1:] = act_arr[i, :-1]
        state21 = np.concatenate([st15_arr[i, :, :3], tq, st15_arr[i, :, 3:]], axis=1)

        # D2-A pre-write bound check (plan 2026-05-21_0545): compute the same
        # normalized deltas demo_loader.py would, reject any demo whose peak
        # exceeds ARGS.bound_max_normalized (strict 1.0, tighter than loader's
        # 1.05 cap). Symmetric prev-action convention: prev[0] = q_init[i].
        qs = q_init[i].astype(np.float32)
        prev = np.vstack([qs[None, :], act_arr[i, :-1]])
        deltas_norm = (act_arr[i] - prev) / BOUND_DIVISOR
        peak_norm = float(np.abs(deltas_norm).max())
        if peak_norm > ARGS.bound_max_normalized:
            print(f"    DROP demo (cube=({cube_p[i,0]:+.3f},{cube_p[i,1]:+.3f}), "
                  f"color={COLOR_NAMES[color]}): peak |normalized| = {peak_norm:.3f} "
                  f"> {ARGS.bound_max_normalized}")
            continue

        demo_idx = len(demos)
        demos.append(dict(
            rgb=rgb_arr[i], state=state21, actions=act_arr[i],
            rewards=rew_arr[i], terminals=done_arr[i],
            color_idx=int(color), cube_pos=cube_p[i].tolist(),
            bowl_pos=bowl_p[i].tolist(), seed=int(seed),
            return_sum=float(rew_arr[i].sum()),
        ))
        color_counts[color] += 1
        n_added += 1

        # Save 1 sanity video per color (use first successful demo of that color)
        if record_hires and color not in saved_video_for_color:
            cname = COLOR_NAMES[color]
            out_mp4 = os.path.join(OUT_DIR, f'sanity_{cname}_demo{demo_idx:03d}.mp4')
            H, W = hires_buf[0].shape[1], hires_buf[0].shape[2]
            proc = subprocess.Popen(
                ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
                 '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                 '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart', out_mp4],
                stdin=subprocess.PIPE)
            for bf in hires_buf:
                proc.stdin.write(cv2.cvtColor(bf[i], cv2.COLOR_RGB2BGR).tobytes())
            proc.stdin.close(); proc.wait()
            saved_video_for_color.add(color)
            print(f"    sanity video → {out_mp4}")

        if len(demos) >= NUM_DEMOS_TARGET: break

    print(f"  batch {batch_i:2d}  seed={seed:5d}  +{n_added}  total {len(demos):2d}/{NUM_DEMOS_TARGET}  "
          f"per_color={color_counts}  ({time.time()-t0:.0f}s)")

env.close()
print(f"\nFinal: {len(demos)} demos in {time.time()-t0:.0f}s")
print(f"Per-color: {color_counts}")
print(f"Sanity videos saved for colors: {sorted(saved_video_for_color)}")

# ─── Write HDF5 v2 schema ───────────────────────────────────────────────────
try:
    commit = subprocess.check_output(['git','rev-parse','--short','HEAD'],
                                      cwd='/home/shadeform/squint').decode().strip()
except Exception:
    commit = 'unknown'

print(f"\nWriting {OUT_H5}...")
with h5py.File(OUT_H5, 'w') as f:
    f.attrs['format_version']    = '2.0'
    f.attrs['env_id']            = 'SO101PlaceCube-v1'
    f.attrs['control_mode']      = 'pd_joint_pos'
    f.attrs['n_distractors']     = 0
    f.attrs['use_real_bowl']     = True
    f.attrs['domain_randomization'] = False
    f.attrs['apply_jitter']      = True
    f.attrs['rgb_h']             = IMG_H
    f.attrs['rgb_w']             = IMG_W
    f.attrs['state_dim']         = 21
    f.attrs['action_dim']        = 6
    f.attrs['arm_delta_max']     = 0.05
    f.attrs['grip_delta_max']    = 0.20
    f.attrs['num_demos']         = len(demos)
    f.attrs['num_colors']        = NUM_COLORS
    f.attrs['T']                 = demos[0]['actions'].shape[0] if demos else 0
    f.attrs['reward_v_min']      = -20.0
    f.attrs['reward_v_max']      = 20.0
    f.attrs['collector_commit']  = commit
    f.attrs['collected_at_utc']  = datetime.datetime.utcnow().isoformat() + 'Z'
    f.attrs['collector_deviation_control_mode'] = (
        "Spec called for pd_joint_target_delta_pos but it has a target-accumulation bug "
        "under cube/finger collision. Actions in this file are ABSOLUTE joint targets (rad). "
        "Convert at load time: delta[t] = action[t] - action[t-1] with action[-1]=QPOS_START, "
        "then normalize by [0.05]*5+[0.2]."
    )
    f.attrs['collector_deviation_msvecenv'] = (
        "Spec called for ManiSkillVectorEnv wrapper but A/B test showed it kills grasp "
        "dynamics. Demos collected without it. Each demo is one uninterrupted episode."
    )
    f.attrs['collector_deviation_dr'] = (
        "Spec called for domain_randomization=True. DR drops scripted-IK grasp rate <5%. "
        "Demos collected with DR=False; trainer can apply per-step visual jitter at training time."
    )
    f.attrs['state_layout'] = 'bowl_xyz_robot_frame(3) + target_qpos(6, synthesized=action[t-1]) + goal_color_onehot(6) + noisy_qpos(6)'

    for i, d in enumerate(demos):
        g = f.create_group(f'demo_{i:03d}')
        g.create_dataset('obs/rgb',   data=d['rgb'],     compression='gzip', compression_opts=4)
        g.create_dataset('obs/state', data=d['state'],   compression='gzip', compression_opts=4)
        g.create_dataset('actions',   data=d['actions'], compression='gzip', compression_opts=4)
        g.create_dataset('rewards',   data=d['rewards'])
        g.create_dataset('terminals', data=d['terminals'])
        g.attrs['color_idx']  = d['color_idx']
        g.attrs['cube_pos']   = d['cube_pos']
        g.attrs['bowl_pos']   = d['bowl_pos']
        g.attrs['seed']       = d['seed']
        g.attrs['success']    = True
        g.attrs['return_sum'] = d['return_sum']

sz = os.path.getsize(OUT_H5)/1e6
print(f"Wrote {len(demos)} demos → {OUT_H5}  ({sz:.1f} MB)")

with open(OUT_META,'w') as f:
    json.dump(dict(
        num_demos=len(demos),
        per_color_counts=color_counts,
        seeds_used=[d['seed'] for d in demos],
        state_dim=21, action_dim=6,
        control_mode='pd_joint_pos',
        rgb_resolution=[IMG_H, IMG_W],
        sanity_videos_for_colors=sorted(saved_video_for_color),
        notes='50-demo collection using V3-style env (no MSVecEnv, no DR). '
              'Same v2 schema as 2-demo pipeline test. Deviation flags in HDF5 attrs.',
    ), f, indent=2)
print(f"Meta → {OUT_META}")
