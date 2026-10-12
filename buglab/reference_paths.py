"""Deterministic reference genomes used only to validate map feasibility."""

from math import atan2, cos, hypot, pi, sin

import numpy as np

from .environment import EpisodeTrace, World, simulate
from .maps import RECTANGULAR_SPIRAL


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def waypoint_reference_genome(world: World) -> np.ndarray:
    """Encode waypoint steering as a fixed open-loop genome under real physics."""
    x, y = world.start
    heading = atan2(world.goal[1] - y, world.goal[0] - x)
    targets = (*world.checkpoints, world.goal)
    target_index = 0
    turns: list[float] = []
    for _ in range(world.steps):
        while target_index < len(targets) - 1 and hypot(
            x - targets[target_index][0], y - targets[target_index][1]
        ) <= 2.0:
            target_index += 1
        target_x, target_y = targets[target_index]
        desired = atan2(target_y - y, target_x - x)
        turn = float(np.clip(_wrap_angle(desired - heading),
                             -world.max_turn, world.max_turn))
        turns.append(turn)
        heading += turn
        next_x = x + world.step_length * cos(heading)
        next_y = y + world.step_length * sin(heading)
        if not world.blocked_scalar(x, y, next_x, next_y):
            x, y = next_x, next_y
    return np.asarray(turns)


def rectangular_spiral_reference() -> tuple[World, np.ndarray, EpisodeTrace]:
    """Return the validated teaching reference; it is never optimizer input."""
    world = World.from_map(RECTANGULAR_SPIRAL)
    genome = waypoint_reference_genome(world)
    return world, genome, simulate(world, genome)
