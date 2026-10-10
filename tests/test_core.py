import numpy as np
from buglab.environment import World,Rectangle,evaluate,simulate
from buglab.experiment import run_method
from buglab.fitness import FitnessResult
from buglab.maps import MapDefinition
from buglab.optimizers import SearchConfig,GeneticAlgorithm,RandomSearch,LocalSearch

def test_determinism():
    world=World(); genome=np.zeros(world.steps)
    a=evaluate(world,genome);b=evaluate(world,genome)
    assert a.fitness==b.fitness
    np.testing.assert_array_equal(a.positions,b.positions)


def test_fixed_seed_rollout_regression():
    """Task 1 baseline: refactoring must not alter the original default episode."""
    world = World()
    genome = np.random.default_rng(2025).normal(0.0, 0.2, world.steps)

    result = evaluate(world, genome)

    assert result.fitness == -151.03848480983496
    assert result.reached is False
    assert result.reached_at is None
    assert result.collisions == 31
    assert result.min_geodesic == 48.899494936611646
    np.testing.assert_allclose(result.positions[-1], [85.07456088102823, 13.855709516418198])
    np.testing.assert_allclose(result.positions.sum(axis=0), [5064.970314342564, 615.6249932737279])

def test_collision_segment_not_just_endpoint():
    rect=Rectangle(3,3,4,4)
    assert rect.hits_segment(np.array([0.,3.5]),np.array([7.,3.5]))
    assert not rect.hits_segment(np.array([0.,5.]),np.array([7.,5.]))


def test_simulation_clamps_relative_turn_angles():
    world = World(width=20, height=20, start=(5, 5), goal=(15, 5), obstacles=(),
                  radius=0.1, goal_radius=0.0, step_length=1.0, steps=1, max_turn=0.4)
    clamped = simulate(world, np.array([world.max_turn]))
    excessive = simulate(world, np.array([100.0]))
    np.testing.assert_array_equal(excessive.positions, clamped.positions)


def test_boundary_blocks_movement_but_consumes_turn_and_step():
    world = World(width=10, height=10, start=(8, 5), goal=(9, 5), obstacles=(),
                  radius=1.0, goal_radius=0.0, step_length=2.0, steps=1)
    result = simulate(world, np.zeros(1))
    assert result.collisions == 1
    assert result.positions.shape == (2, 2)
    np.testing.assert_array_equal(result.positions[0], result.positions[1])


def test_simulation_prevents_tunneling_through_obstacle():
    world = World(width=20, height=10, start=(2, 5), goal=(18, 5),
                  obstacles=(Rectangle(6, 4, 6.2, 6),), radius=0.1,
                  goal_radius=0.0, step_length=8.0, steps=1)
    result = simulate(world, np.zeros(1))
    assert result.collisions == 1
    np.testing.assert_array_equal(result.positions, [[2.0, 5.0], [2.0, 5.0]])


def test_map_and_fitness_are_pluggable_without_changing_evaluate_contract():
    definition = MapDefinition(12, 8, (2, 4), (10, 4), ())
    world = World.from_map(definition, steps=1, step_length=1.0)

    class ConstantFitness:
        def score(self, scored_world, trace):
            assert scored_world is world
            return FitnessResult(12.5, 7.0)

    result = evaluate(world, np.zeros(1), ConstantFitness())
    assert result.fitness == 12.5
    assert result.min_geodesic == 7.0

    cfg = SearchConfig(genome_length=1, population=2, elite=1)
    run = run_method(world, cfg, "random", seed=3, generations=1,
                     snapshot_generations=(0,), fitness=ConstantFitness())
    assert run.best_rollout.fitness == 12.5


def test_snapshots_store_replayable_bounded_candidate_batches():
    world = World(steps=4)
    cfg = SearchConfig(genome_length=4, population=5, elite=1)
    run = run_method(world, cfg, "random", seed=7, generations=3,
                     snapshot_generations=(1,))
    assert [snapshot.generation for snapshot in run.snapshots] == [1, 2]
    for snapshot in run.snapshots:
        assert snapshot.candidate_genomes.shape == (5, 4)
        replayed = [evaluate(world, genome).fitness for genome in snapshot.candidate_genomes]
        np.testing.assert_array_equal(replayed, snapshot.candidate_scores)

def test_optimizers_contract():
    cfg=SearchConfig(genome_length=8,population=10,elite=1)
    for method in (GeneticAlgorithm,RandomSearch,LocalSearch):
        agent=method(cfg,0)
        for _ in range(3):
            x=agent.ask();assert x.shape==(10,8)
            assert np.all(np.abs(x)<cfg.max_turn+1e-12)
            agent.tell(x,-(x*x).sum(axis=1))
