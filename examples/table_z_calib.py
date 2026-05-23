"""Calibrate the table plane in FK-z as a 2D field z_table(x, y).

The grasp gate/nudge/place assume FK z=0 is the table everywhere, but the real
arm's geometry diverges from the URDF with pose, so the FK z that actually means
"touching the table" drifts across the workspace — not just with reach r, but
with lateral position too. This records it in 2D: with the gripper held CLOSED
and arm torque OFF, you slide the closed fingertip all over the table — sweep
near→far AND left→right (cover the area you'll grasp in). It logs (x, y, z) at
the TCP, bins them into a voxel grid, fills gaps by interpolation, and saves a
dense z_table(x, y) grid to table_z_calib.json.

infer_linux.table_z(x, y) bilinearly interpolates that grid (a radial line a·r+b
is also fit and stored as a fallback for older single-sweep calibs).

    python examples/table_z_calib.py                 # default /dev/ttyACM0
    python examples/table_z_calib.py --port /dev/ttyUSB0 --cell_cm 1.0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy_utils import infer_linux
from deploy_utils.infer_linux import create_real_robot
from deploy_utils.so101_fk import tcp_pos
from lerobot.motors.motors_bus import MotorNormMode

G_SIM_MIN, G_SIM_MAX = -10.0, 120.0
G_SRV_MIN, G_SRV_MAX = -60.13, 66.73
CLOSE_SIM = -10.0
CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "table_z_calib.json")


def srv_to_sim(s):
    return (s - G_SRV_MIN) / (G_SRV_MAX - G_SRV_MIN) * (G_SIM_MAX - G_SIM_MIN) + G_SIM_MIN


def sim_to_srv(s):
    return (s - G_SIM_MIN) / (G_SIM_MAX - G_SIM_MIN) * (G_SRV_MAX - G_SRV_MIN) + G_SRV_MIN


def build_grid(xs, ys, zs, cell):
    """Voxel-grid z_table(x,y): median z per `cell`-sized cell, then fill empty
    cells by interpolation (linear inside the swept hull, nearest outside) so the
    grid is dense. Returns (Z[ny,nx], x_min, y_min, nx, ny)."""
    from scipy.interpolate import griddata
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    nx = int(round((x_max - x_min) / cell)) + 1
    ny = int(round((y_max - y_min) / cell)) + 1
    gx = x_min + np.arange(nx) * cell
    gy = y_min + np.arange(ny) * cell

    # 1) median z of the samples falling in each cell (dwell-time + outlier robust)
    ix = np.clip(np.round((xs - x_min) / cell).astype(int), 0, nx - 1)
    iy = np.clip(np.round((ys - y_min) / cell).astype(int), 0, ny - 1)
    Z = np.full((ny, nx), np.nan)
    for cy in range(ny):
        for cx in range(nx):
            m = (ix == cx) & (iy == cy)
            if m.any():
                Z[cy, cx] = np.median(zs[m])
    n_filled = int(np.isfinite(Z).sum())

    # 2) fill the empty cells from the filled cell centres
    GX, GY = np.meshgrid(gx, gy)
    fy, fx = np.where(np.isfinite(Z))
    pts = np.column_stack([gx[fx], gy[fy]])
    vals = Z[fy, fx]
    znear = griddata(pts, vals, (GX, GY), method="nearest")          # always defined
    try:
        zlin = griddata(pts, vals, (GX, GY), method="linear")        # NaN outside hull / if collinear
        Zdense = np.where(np.isfinite(zlin), zlin, znear)
    except Exception:
        Zdense = znear                                                # collinear sweep → nearest only
    return Zdense, x_min, y_min, nx, ny, n_filled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=str, default=infer_linux.ROBOT_PORT,
                   help=f"serial device (default {infer_linux.ROBOT_PORT})")
    p.add_argument("--cell_cm", type=float, default=1.0, help="voxel grid cell size (cm; default 1.0)")
    args = p.parse_args()
    infer_linux.ROBOT_PORT = args.port
    cell = args.cell_cm / 100.0

    robot = create_real_robot()
    robot.connect()
    bus = robot.bus
    bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    present = bus.sync_read("Present_Position")
    keys = list(present.keys())                       # bus order == FK joint order
    arm_keys = [k for k in keys if k != "gripper"]

    def qpos_rad():
        d = bus.sync_read("Present_Position")
        deg = [srv_to_sim(d[k]) if k == "gripper" else d[k] for k in keys]
        return np.deg2rad(np.array(deg, dtype=np.float64))

    # Close the gripper (torque on) so the contact point is a stable single tip,
    # then free the arm so it can be slid by hand.
    cmd = {f"{k}.pos": float(present[k]) for k in keys}
    cmd["gripper.pos"] = float(sim_to_srv(CLOSE_SIM))
    robot.send_action(cmd)
    time.sleep(1.5)
    bus.disable_torque(arm_keys)

    print("\nArm is now FREE (gripper held closed).")
    print("Keep the closed fingertip touching the table and slide it slowly to")
    print("cover the whole workspace — near↔far AND left↔right. Press Ctrl+C when done.\n")

    xs, ys, zs = [], [], []
    try:
        while True:
            tcp = tcp_pos(qpos_rad())
            xs.append(float(tcp[0])); ys.append(float(tcp[1])); zs.append(float(tcp[2]))
            print(f"  samples={len(xs):5d}  x={tcp[0]*100:6.1f}  y={tcp[1]*100:6.1f}  "
                  f"z={tcp[2]*100:6.2f} cm   ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    print(f"\n\nCollected {len(xs)} samples, "
          f"x ∈ [{xs.min()*100:.1f}, {xs.max()*100:.1f}] cm, "
          f"y ∈ [{ys.min()*100:.1f}, {ys.max()*100:.1f}] cm")
    if len(xs) < 20:
        print("Too few samples — not saving. Re-run and sweep for longer.")
        bus.enable_torque(arm_keys); time.sleep(0.5); robot.disconnect()
        return

    # Radial line fallback (z ≈ a·r + b), median over 1 cm r-bins — kept so the
    # grid has a sane 1D backstop and older loaders still work.
    rs = np.hypot(xs, ys)
    edges = np.arange(rs.min(), rs.max() + 0.01, 0.01)
    idx = np.digitize(rs, edges)
    bin_r, bin_z = [], []
    for bi in np.unique(idx):
        m = idx == bi
        bin_r.append(float(rs[m].mean())); bin_z.append(float(np.median(zs[m])))
    a, b = (np.polyfit(bin_r, bin_z, 1) if len(bin_r) >= 2 else (0.0, float(np.median(zs))))

    # 2D voxel grid.
    Z, x_min, y_min, nx, ny, n_filled = build_grid(xs, ys, zs, cell)
    # RMSE of the grid vs the raw samples (bilinear-sample the grid at each point).
    def bilerp(x, y):
        fx = min(max((x - x_min) / cell, 0.0), nx - 1.0); fy = min(max((y - y_min) / cell, 0.0), ny - 1.0)
        x0, y0 = int(fx), int(fy); x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
        tx, ty = fx - x0, fy - y0
        top = Z[y0, x0] * (1 - tx) + Z[y0, x1] * tx
        bot = Z[y1, x0] * (1 - tx) + Z[y1, x1] * tx
        return top * (1 - ty) + bot * ty
    rmse = float(np.sqrt(np.mean([(bilerp(x, y) - z) ** 2 for x, y, z in zip(xs, ys, zs)])))

    np.savez(os.path.splitext(CALIB_PATH)[0] + "_raw.npz", x=xs, y=ys, z=zs)
    out = {
        "mode": "grid",
        "cell_m": cell,
        "x_min": x_min, "y_min": y_min, "nx": nx, "ny": ny,
        "x_max": float(xs.max()), "y_max": float(ys.max()),
        "z": Z.tolist(),
        "a": float(a), "b": float(b),          # radial line fallback
        "n": int(len(xs)), "n_cells_filled": n_filled, "rmse_m": rmse,
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(out, f)
    print(f"2D grid: {nx}×{ny} cells @ {args.cell_cm:.1f} cm "
          f"({n_filled}/{nx*ny} directly measured, rest interpolated), grid RMSE {rmse*100:.2f} cm")
    print(f"  radial fallback z[m] = {a:.4f}·r + {b:.4f}")
    print(f"  → saved {CALIB_PATH}  (+ raw samples in *_raw.npz)")

    bus.enable_torque(arm_keys)
    time.sleep(0.5)
    robot.disconnect()


if __name__ == "__main__":
    main()
