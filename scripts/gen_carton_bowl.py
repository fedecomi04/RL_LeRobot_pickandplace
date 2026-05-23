"""Generate a white-carton bowl mesh (truncated-cone shell) for the place task.

Spec (from the real cardboard bowl):
  - lower (bottom) diameter 10 cm  -> outer bottom radius 0.050 m
  - upper (rim)    diameter 15 cm  -> outer rim    radius 0.075 m
  - height                4.5 cm   -> 0.045 m
  - wall/floor thickness  0.1 cm   -> 0.001 m
  - floor<->wall junction ROUNDED with a 3 cm fillet (no sharp crease)

The floor and the conical wall are joined by a 1 cm-radius fillet (tangent to
both), so there is no sharp edge between the base and the contour — like a real
moulded paper bowl. Built as a watertight solid of revolution from a closed 2D
(r, z) profile so CoACD can decompose it into convex hulls for dynamic collision
while keeping the open cavity. Origin at bowl bottom-center (floor underside on
z=0). Writes envs/meshes/bowl_carton.obj.
"""
import os
import numpy as np
import trimesh

H   = 0.045   # height
T   = 0.001   # thickness (walls + floor)
RBO = 0.050   # nominal outer bottom radius (defines wall slope)
RTO = 0.075   # outer rim radius (upper diameter 15 cm)
RF  = 0.030   # fillet radius at the base<->wall junction (3 cm)
SECTIONS = 96 # angular resolution
ARC_N = 16    # points per fillet arc


def arc_pts(center, radius, a0, a1, n):
    """Points on a circle (center=(r,z)) from angle a0 to a1 (radians)."""
    ang = np.linspace(a0, a1, n)
    return np.stack([center[0] + radius * np.cos(ang),
                     center[1] + radius * np.sin(ang)], axis=1)


# Wall slope: cone half-angle from vertical, set by the nominal bottom/rim radii.
alpha = np.arctan2(RTO - RBO, H)
sa, ca = np.sin(alpha), np.cos(alpha)

# Outer wall line through the rim A=(RTO, H): ca*r - sa*z + c0 = 0.
c0 = -(ca * RTO - sa * H)
# Fillet centre C=(rc, RF): tangent to the floor plane z=0 (=> centre z=RF) and
# tangent to the outer wall line on the interior side (signed dist = -RF).
rc = (-RF + sa * RF - c0) / ca
C = np.array([rc, RF])

# Tangent angles on the fillet: floor tangent points straight down (-90 deg);
# wall tangent is along the wall's outward normal n=(ca,-sa).
a_floor = -np.pi / 2.0
a_wall = np.arctan2(-sa, ca)

A_out = np.array([RTO, H])              # outer rim
A_in = A_out - T * np.array([ca, -sa])  # inner rim (offset inward by thickness)

# ── Closed profile, axis-bottom -> outer -> rim -> inner -> axis-cavity ──────
# Arc endpoints already land on the flat-floor tangent points (rc, 0) / (rc, T),
# so we do NOT repeat them (duplicates make degenerate, non-watertight faces).
prof = [[0.0, 0.0]]                                  # axis, floor underside
prof += arc_pts(C, RF, a_floor, a_wall, ARC_N).tolist()   # outer floor flat start + fillet
prof += [A_out.tolist()]                             # up the outer wall to rim
prof += [A_in.tolist()]                              # across the rim (top face)
prof += arc_pts(C, RF - T, a_wall, a_floor, ARC_N).tolist()  # inner wall + fillet -> (rc, T)
prof += [[0.0, T]]                                   # cavity floor to axis
profile = np.array(prof, dtype=np.float64)


def lathe(profile, sections):
    """Revolve a closed (r, z) profile around the z-axis into a watertight mesh.
    Axis points (r==0) become a single shared vertex; segments touching the axis
    become triangle fans, the rest quad strips."""
    M = len(profile)
    thetas = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    cos, sin = np.cos(thetas), np.sin(thetas)
    verts, vid = [], [None] * M
    for j, (r, z) in enumerate(profile):
        if r <= 1e-9:
            verts.append([0.0, 0.0, z]); vid[j] = len(verts) - 1
        else:
            ids = []
            for s in range(sections):
                verts.append([r * cos[s], r * sin[s], z]); ids.append(len(verts) - 1)
            vid[j] = np.array(ids)
    faces = []
    for j in range(M):
        a, b = j, (j + 1) % M
        ra, rb = profile[a, 0], profile[b, 0]
        if ra <= 1e-9 and rb <= 1e-9:
            continue
        for s in range(sections):
            sn = (s + 1) % sections
            if ra <= 1e-9:
                faces.append([vid[a], vid[b][s], vid[b][sn]])
            elif rb <= 1e-9:
                faces.append([vid[a][s], vid[b], vid[a][sn]])
            else:
                faces.append([vid[a][s], vid[b][s], vid[b][sn]])
                faces.append([vid[a][s], vid[b][sn], vid[a][sn]])
    mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


bowl = lathe(profile, SECTIONS)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "envs", "meshes", "bowl_carton.obj")
bowl.export(out)

mn, mx = bowl.bounds
print(f"watertight={bowl.is_watertight}  volume={bowl.volume*1e6:.2f} cm^3")
print(f"AABB: x[{mn[0]:.4f},{mx[0]:.4f}] z[{mn[2]:.4f},{mx[2]:.4f}]")
print(f"rim Ø={2*RTO*100:.1f}cm  height={H*100:.1f}cm  thickness={T*100:.2f}cm  fillet R={RF*100:.1f}cm")
print(f"fillet centre r={rc*100:.2f}cm  flat-floor radius={rc*100:.2f}cm")
print(f"wrote {out}  ({len(bowl.vertices)} verts, {len(bowl.faces)} faces)")
