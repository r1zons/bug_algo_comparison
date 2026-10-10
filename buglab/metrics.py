"""Reward-independent outcome metrics for beetle episodes."""

from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .environment import EpisodeTrace, World


@dataclass(frozen=True)
class OutcomeMetrics:
    """Physical outcomes that remain comparable across fitness strategies."""

    goal_reached: bool
    steps_to_goal: int | None
    collision_count: int
    actual_path_length: float
    final_distance: float
    environment_steps: int


def outcome_metrics(world: "World", trace: "EpisodeTrace") -> OutcomeMetrics:
    """Calculate shared metrics without consulting the fitness policy."""
    displacements = np.diff(trace.positions, axis=0)
    path_length = float(np.linalg.norm(displacements, axis=1).sum())
    final_x, final_y = trace.positions[-1]
    goal_x, goal_y = world.goal
    return OutcomeMetrics(
        goal_reached=trace.reached,
        steps_to_goal=trace.reached_at,
        collision_count=trace.collisions,
        actual_path_length=path_length,
        final_distance=hypot(float(final_x) - goal_x, float(final_y) - goal_y),
        environment_steps=len(trace.positions) - 1,
    )
