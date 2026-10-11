"""Explicit fitness policies composed with deterministic beetle physics."""

from dataclasses import dataclass
from math import hypot
from typing import Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .environment import EpisodeTrace, World


@dataclass(frozen=True)
class FitnessResult:
    """Scalar training signal and its closest-distance diagnostic."""

    fitness: float
    min_distance: float


class FitnessPolicy(Protocol):
    """A scoring policy; optimizers still receive only ``fitness``."""

    name: str

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult: ...


def _euclidean_distances(world: "World", trace: "EpisodeTrace") -> np.ndarray:
    return np.linalg.norm(trace.positions - np.asarray(world.goal), axis=1)


@dataclass(frozen=True)
class EuclideanFitness:
    """Reward final or best straight-line proximity; it knows no obstacles."""

    use_best: bool = True
    name: str = "euclidean"

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        distances = _euclidean_distances(world, trace)
        distance = float(distances.min() if self.use_best else distances[-1])
        return FitnessResult(-distance, float(distances.min()))


@dataclass(frozen=True)
class SparseGoalFitness:
    """Reward physical goal arrival only; distance is diagnostic, not reward."""

    name: str = "sparse_goal"

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        closest = float(_euclidean_distances(world, trace).min())
        return FitnessResult(1.0 if trace.reached else 0.0, closest)


@dataclass(frozen=True)
class GeodesicFitness:
    """Original obstacle-aware shaping, including success and collision terms.

    The distance grid exposes map structure to the reward. It is a training
    signal, not a route supplied to an optimizer.
    """

    distance_weight: float = 3.0
    collision_weight: float = 0.14
    success_bonus: float = 260.0
    time_weight: float = 0.45
    name: str = "geodesic"

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        closest = min(world.geodesic_xy(float(x), float(y)) for x, y in trace.positions)
        fitness = -self.distance_weight * closest - self.collision_weight * trace.collisions
        if trace.reached:
            assert trace.reached_at is not None
            fitness += self.success_bonus - self.time_weight * trace.reached_at
        return FitnessResult(float(fitness), float(closest))


@dataclass(frozen=True)
class SuccessDistanceFitness:
    """Rank every physical success above every failure, then prefer speed."""

    name: str = "success_distance"

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        closest = float(_euclidean_distances(world, trace).min())
        if trace.reached:
            assert trace.reached_at is not None
            # At least 1.0 for a final-step success; every failure is negative.
            score = float(world.steps + 1 - trace.reached_at)
        else:
            score = -closest
        return FitnessResult(score, closest)


@dataclass(frozen=True)
class CollisionPenaltyFitness:
    """Compose a configurable collision penalty with another base reward."""

    base: FitnessPolicy = EuclideanFitness()
    penalty: float = 0.5
    name: str = "collision_penalty"

    def __post_init__(self) -> None:
        if self.penalty < 0:
            raise ValueError("Collision penalty must be non-negative")

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        result = self.base.score(world, trace)
        return FitnessResult(result.fitness - self.penalty * trace.collisions, result.min_distance)


@dataclass(frozen=True)
class CheckpointFitness:
    """Reward ordered map-specific checkpoints, then proximity to the next one.

    Checkpoints encode designer knowledge about a useful corridor and therefore
    are not information-neutral relative to Euclidean shaping.
    """

    checkpoint_radius: float = 3.0
    checkpoint_reward: float = 100.0
    name: str = "checkpoints"

    def __post_init__(self) -> None:
        if self.checkpoint_radius <= 0 or self.checkpoint_reward <= 0:
            raise ValueError("Checkpoint radius and reward must be positive")

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        targets = (*world.checkpoints, world.goal)
        completed = 0
        for position in trace.positions:
            if completed >= len(targets):
                break
            target = targets[completed]
            if hypot(float(position[0]) - target[0], float(position[1]) - target[1]) <= self.checkpoint_radius:
                completed += 1

        if completed == len(targets):
            remaining = 0.0
        else:
            target = np.asarray(targets[completed])
            remaining = float(np.linalg.norm(trace.positions - target, axis=1).min())
        closest_goal = float(_euclidean_distances(world, trace).min())
        return FitnessResult(completed * self.checkpoint_reward - remaining, closest_goal)


@dataclass(frozen=True)
class SmoothnessFitness:
    """Compose a turn-change penalty with a configurable base reward."""

    base: FitnessPolicy = EuclideanFitness()
    weight: float = 1.0
    name: str = "smoothness"

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("Smoothness weight must be non-negative")

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        result = self.base.score(world, trace)
        turn_changes = np.diff(trace.turns)
        penalty = self.weight * float(np.square(turn_changes).sum())
        return FitnessResult(result.fitness - penalty, result.min_distance)


@dataclass(frozen=True)
class LegacyFitness:
    """Compatibility policy for historical ``World(fitness_mode=...)`` calls."""

    mode: str = "geodesic"
    name: str = "legacy"

    def __post_init__(self) -> None:
        if self.mode not in ("geodesic", "euclidean"):
            raise ValueError("fitness mode must be geodesic or euclidean")

    def score(self, world: "World", trace: "EpisodeTrace") -> FitnessResult:
        if self.mode == "geodesic":
            return GeodesicFitness().score(world, trace)
        closest = float(_euclidean_distances(world, trace).min())
        fitness = -3.0 * closest - 0.14 * trace.collisions
        if trace.reached:
            assert trace.reached_at is not None
            fitness += 260.0 - 0.45 * trace.reached_at
        return FitnessResult(float(fitness), closest)


FITNESS_NAMES = (
    "geodesic",
    "euclidean",
    "sparse_goal",
    "success_distance",
    "collision_penalty",
    "checkpoints",
    "smoothness",
)


def make_fitness(
    name: str,
    *,
    collision_penalty: float = 0.5,
    smoothness_weight: float = 1.0,
    base: FitnessPolicy | None = None,
) -> FitnessPolicy:
    """Build a named policy with configurable composition weights."""
    if name == "geodesic":
        return GeodesicFitness()
    if name == "euclidean":
        return EuclideanFitness()
    if name == "sparse_goal":
        return SparseGoalFitness()
    if name == "success_distance":
        return SuccessDistanceFitness()
    if name == "collision_penalty":
        return CollisionPenaltyFitness(base if base is not None else EuclideanFitness(), collision_penalty)
    if name == "checkpoints":
        return CheckpointFitness()
    if name == "smoothness":
        return SmoothnessFitness(base if base is not None else EuclideanFitness(), smoothness_weight)
    raise ValueError(f"Unknown fitness {name!r}; expected one of {FITNESS_NAMES}")
