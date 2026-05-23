"""Reusable pick-and-place for the SO101, for use across evals.

The picking policy + FK-gated hardcoded grasp (solved 2026-05-20) plus an IK
place phase, wrapped in one callable:

    from final_utils import pick_and_place
    ok = pick_and_place(goal_color=0, bowl_xy=(0.25, 0.10))

EVAL 2 (two cubes) splits the cubes apart first, then runs the same pick-and-place:

    from final_utils import split_pick_place
    ok = split_pick_place(goal_color=0, bowl_xy=(0.25, 0.20))

EVAL 3 (four cubes) splits them, then picks three queried colours IN ORDER:

    from final_utils import split_pick_place_sequence
    ok = split_pick_place_sequence(goal_colors=(0, 2, 4), bowl_xy=(0.25, 0.20))

See final_utils/pick_place.py, final_utils/eval2.py and final_utils/eval3.py.
"""
from .pick_place import pick_and_place
from .eval2 import split_pick_place
from .eval3 import split_pick_place_sequence

__all__ = ["pick_and_place", "split_pick_place", "split_pick_place_sequence"]
