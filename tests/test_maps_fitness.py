import csv
from math import atan2, cos, hypot, pi, sin

import numpy as np

from buglab.environment import EpisodeTrace, World, evaluate, simulate
from buglab.experiment import run_method, save_history, save_outcome_comparison
from buglab.fitness import (
    CheckpointFitness,
    CollisionPenaltyFitness,
    EuclideanFitness,
    GeodesicFitness,
    SmoothnessFitness,
    SparseGoalFitness,
    SuccessDistanceFitness,
)
from buglab.maps import MAPS, SPIRAL
from buglab.metrics import outcome_metrics
from buglab.optimizers import SearchConfig
from buglab.visualization import map_gallery


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def reference_genome(world: World) -> np.ndarray:
    """Encode deterministic waypoint steering as open-loop relative turns."""
    x, y = world.start
    goal_x, goal_y = world.goal
    heading = atan2(goal_y - y, goal_x - x)
    targets = [*world.checkpoints, world.goal]
    target_index = 0
    turns = []
    for _ in range(world.steps):
        while (target_index < len(targets) - 1 and
               hypot(x - targets[target_index][0], y - targets[target_index][1]) <= 2.0):
            target_index += 1
        target_x, target_y = targets[target_index]
        desired = atan2(target_y - y, target_x - x)
        turn = float(np.clip(_wrap_angle(desired - heading), -world.max_turn, world.max_turn))
        turns.append(turn)
        heading += turn
        next_x = x + world.step_length * cos(heading)
        next_y = y + world.step_length * sin(heading)
        if not world.blocked_scalar(x, y, next_x, next_y):
            x, y = next_x, next_y
    return np.asarray(turns)


def _trace(positions, *, reached=False, reached_at=None, collisions=0, turns=None):
    positions = np.asarray(positions, dtype=float)
    if turns is None:
        turns = np.zeros(len(positions) - 1)
    return EpisodeTrace(positions, reached, reached_at, collisions, np.asarray(turns, dtype=float))


def test_all_named_maps_are_valid_and_reachable_with_real_physics():
    assert tuple(MAPS) == (
        "default", "open_field", "two_bridges", "spiral", "rectangular_spiral",
        "narrow_gate", "dead_end"
    )
    for definition in MAPS.values():
        definition.validate_clearance(radius=1.1)
        world = World.from_map(definition)
        assert world.geodesic_xy(*world.start) < 500.0
        result = simulate(world, reference_genome(world))
        assert result.reached, definition.name
        assert result.reached_at <= world.steps
        assert result.collisions == 0


def test_spiral_direct_euclidean_route_is_blocked_but_corridor_route_reaches():
    world = World.from_map(SPIRAL)
    direct = simulate(world, np.zeros(world.steps))
    corridor = simulate(world, reference_genome(world))
    assert not direct.reached
    assert direct.collisions > 0
    assert corridor.reached
    assert GeodesicFitness().score(world, corridor).fitness > GeodesicFitness().score(world, direct).fitness
    assert CheckpointFitness().score(world, corridor).fitness > CheckpointFitness().score(world, direct).fitness


def test_euclidean_supports_best_and_final_proximity():
    world = World(width=20, height=20, start=(2, 10), goal=(12, 10), obstacles=(), steps=2)
    trace = _trace([(2, 10), (10, 10), (7, 10)])
    assert EuclideanFitness(use_best=True).score(world, trace).fitness == -2.0
    assert EuclideanFitness(use_best=False).score(world, trace).fitness == -5.0


def test_success_distance_always_prioritizes_physical_success():
    world = World(width=20, height=20, start=(2, 10), goal=(12, 10), obstacles=(), steps=3)
    success = _trace([(2, 10), (6, 10), (10, 10)], reached=True, reached_at=2)
    near_failure = _trace([(2, 10), (9.9, 10), (10.1, 10)])
    policy = SuccessDistanceFitness()
    assert policy.score(world, success).fitness > policy.score(world, near_failure).fitness


def test_sparse_goal_reward_depends_only_on_physical_arrival():
    world = World(width=20, height=20, start=(2, 10), goal=(12, 10), obstacles=(), steps=3)
    success = _trace([(2, 10), (6, 10), (10, 10)], reached=True, reached_at=2)
    near_failure = _trace([(2, 10), (9.9, 10), (10.1, 10)])
    far_failure = _trace([(2, 10), (3, 10), (4, 10)])
    policy = SparseGoalFitness()
    assert policy.score(world, success).fitness == 1.0
    assert policy.score(world, near_failure).fitness == 0.0
    assert policy.score(world, far_failure).fitness == 0.0
    assert policy.score(world, near_failure).min_distance < policy.score(world, far_failure).min_distance


def test_collision_and_smoothness_are_configurable_compositions():
    world = World(width=20, height=20, start=(2, 10), goal=(12, 10), obstacles=(), steps=3)
    trace = _trace([(2, 10), (3, 10), (3, 10), (4, 10)], collisions=2,
                   turns=[0.0, 0.4, -0.4])
    base = EuclideanFitness()
    base_score = base.score(world, trace).fitness
    assert CollisionPenaltyFitness(base, penalty=2.5).score(world, trace).fitness == base_score - 5.0
    assert SmoothnessFitness(base, weight=2.0).score(world, trace).fitness < base_score
    assert SmoothnessFitness(base, weight=0.0).score(world, trace).fitness == base_score


def test_checkpoints_must_be_reached_in_order():
    world = World(width=20, height=20, start=(2, 10), goal=(18, 10), obstacles=(),
                  checkpoints=((6, 10), (12, 10)), steps=3)
    ordered = _trace([(2, 10), (6, 10), (12, 10), (18, 10)], reached=True, reached_at=3)
    reversed_trace = _trace([(2, 10), (12, 10), (6, 10), (18, 10)], reached=True, reached_at=3)
    policy = CheckpointFitness(checkpoint_radius=1.0)
    assert policy.score(world, ordered).fitness == 300.0
    assert policy.score(world, ordered).fitness > policy.score(world, reversed_trace).fitness


def test_outcome_metrics_are_independent_of_fitness_and_count_actual_motion():
    world = World(width=20, height=20, start=(2, 10), goal=(12, 10), obstacles=(), steps=3)
    trace = _trace([(2, 10), (5, 10), (5, 10), (9, 10)], collisions=1)
    metrics = outcome_metrics(world, trace)
    assert metrics.goal_reached is False
    assert metrics.steps_to_goal is None
    assert metrics.collision_count == 1
    assert metrics.actual_path_length == 7.0
    assert metrics.final_distance == 3.0
    assert metrics.environment_steps == 3


def test_runs_are_reproducible_and_record_shared_metrics():
    world = World.from_map(MAPS["open_field"], steps=12)
    cfg = SearchConfig(genome_length=12, population=6, elite=1)
    first = run_method(world, cfg, "random", seed=11, generations=3,
                       snapshot_generations=(0, 2), fitness=EuclideanFitness())
    second = run_method(world, cfg, "random", seed=11, generations=3,
                        snapshot_generations=(0, 2), fitness=EuclideanFitness())
    np.testing.assert_array_equal(first.best_genome, second.best_genome)
    assert first.history == second.history
    assert first.total_evaluations == 18
    assert first.history[-1]["evaluations"] == 18
    for key in ("goal_reached", "steps_to_goal", "collision_count", "actual_path_length",
                "final_distance", "success_rate", "environment_steps"):
        assert key in first.history[-1]


def test_exports_record_configuration_and_cross_fitness_csv_omits_raw_scores(tmp_path):
    world = World.from_map(MAPS["open_field"], steps=4)
    cfg = SearchConfig(genome_length=4, population=4, elite=1)
    run = run_method(world, cfg, "random", seed=5, generations=2,
                     snapshot_generations=(1,), fitness=EuclideanFitness())
    save_history([run], tmp_path)
    with (tmp_path / "summary.csv").open(newline="") as file:
        summary = next(csv.DictReader(file))
    assert summary["seed"] == "5"
    assert summary["map"] == "open_field"
    assert summary["fitness"] == "euclidean"
    assert summary["population"] == "4"
    assert summary["evaluations"] == "8"

    comparison_path = tmp_path / "comparison.csv"
    save_outcome_comparison([run], comparison_path)
    with comparison_path.open(newline="") as file:
        reader = csv.DictReader(file)
        assert "best_fitness" not in reader.fieldnames
        assert next(reader)["final_distance"]


def test_map_gallery_renders_every_preset(tmp_path):
    path = tmp_path / "maps.png"
    map_gallery(path)
    assert path.is_file()
    assert path.stat().st_size > 0
