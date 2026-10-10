from dataclasses import replace

import numpy as np
import pytest

from buglab.environment import World, evaluate
from buglab.experiment import run_method
from buglab.fitness import GeodesicFitness
from buglab.operator_story import (
    classification_tolerance,
    classify_operation,
    load_operation_log,
    reconstruct_child,
    save_operation_log,
    select_operation_examples,
    synthetic_operation,
)
from buglab.optimizers import GeneticAlgorithm, SearchConfig
from buglab.visualization import real_operation_figure, synthetic_operation_figure


def _complete_one_ga_generation(seed=3, trace_limit=64):
    cfg = SearchConfig(genome_length=12, population=8, elite=2)
    optimizer = GeneticAlgorithm(cfg, seed, trace_operations=True, trace_limit=trace_limit)
    initial = optimizer.ask()
    initial_scores = -np.square(initial).sum(axis=1)
    optimizer.tell(initial, initial_scores)
    children = optimizer.ask()
    child_scores = -np.square(children).sum(axis=1)
    optimizer.tell(children, child_scores)
    return cfg, optimizer, initial, initial_scores, children, child_scores


def test_synthetic_example_is_reproducible_and_reconstructs_exactly():
    first = synthetic_operation()
    second = synthetic_operation()
    assert first.cut == second.cut == 4
    assert first.mutated_indices == second.mutated_indices == (2, 6)
    for name in ("parent_a", "parent_b", "child_before_mutation", "mutation_deltas", "child"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_allclose(
        np.rad2deg(first.parent_a), [-18, -12, -6, 0, 6, 12, 18, 24], atol=1e-12
    )
    np.testing.assert_array_equal(reconstruct_child(first), first.child)


def test_initial_generation_has_no_fabricated_operations():
    cfg = SearchConfig(genome_length=8, population=6, elite=1)
    optimizer = GeneticAlgorithm(cfg, 4, trace_operations=True)
    initial = optimizer.ask()
    optimizer.tell(initial, np.arange(cfg.population, dtype=float))
    assert optimizer.operation_log == []


def test_elites_and_offspring_keep_candidate_score_alignment():
    cfg, optimizer, parents, parent_scores, children, child_scores = _complete_one_ga_generation()
    assert len(optimizer.operation_log) == cfg.population
    elite_ids = np.argsort(parent_scores)[-cfg.elite:][::-1]
    for candidate_index, parent_id in enumerate(elite_ids):
        event = optimizer.operation_log[candidate_index]
        assert event.kind == "elite"
        assert event.parent_a_id == parent_id
        assert event.parent_b_id is None
        assert event.cut is None
        assert event.mutated_indices == ()
        np.testing.assert_array_equal(event.child, parents[parent_id])
        np.testing.assert_array_equal(children[candidate_index], parents[parent_id])
        assert event.child_score == child_scores[candidate_index]

    for event in optimizer.operation_log[cfg.elite:]:
        assert event.kind == "offspring"
        assert event.parent_b is not None
        expected = np.concatenate((event.parent_a[:event.cut], event.parent_b[event.cut:]))
        np.testing.assert_array_equal(event.child_before_mutation, expected)
        np.testing.assert_array_equal(reconstruct_child(event), event.child)
        np.testing.assert_array_equal(event.child, children[event.candidate_index])
        assert event.child_score == child_scores[event.candidate_index]
        assert len(event.child) == cfg.genome_length
        assert np.all(np.abs(event.child) <= cfg.max_turn)


def test_tracing_detects_reordered_ask_tell_candidates():
    cfg = SearchConfig(genome_length=8, population=6, elite=1)
    optimizer = GeneticAlgorithm(cfg, 2, trace_operations=True)
    initial = optimizer.ask()
    optimizer.tell(initial, np.arange(cfg.population, dtype=float))
    children = optimizer.ask()
    with pytest.raises(ValueError, match="candidate order"):
        optimizer.tell(children[::-1], np.arange(cfg.population, dtype=float))


def test_logging_enabled_and_disabled_have_identical_search_results():
    world = World(steps=16)
    cfg = SearchConfig(genome_length=16, population=8, elite=2)
    plain = run_method(world, cfg, "ga", seed=9, generations=5,
                       snapshot_generations=(0, 4), trace_operations=False)
    traced = run_method(world, cfg, "ga", seed=9, generations=5,
                        snapshot_generations=(0, 4), trace_operations=True,
                        operation_trace_limit=64)
    np.testing.assert_array_equal(plain.best_genome, traced.best_genome)
    assert plain.best_rollout.fitness == traced.best_rollout.fitness
    assert plain.history == traced.history
    assert plain.total_evaluations == traced.total_evaluations
    assert plain.operation_log == []
    assert traced.operation_log


def test_real_events_replay_scores_and_outcomes_deterministically():
    world = World(steps=14)
    cfg = SearchConfig(genome_length=14, population=8, elite=2)
    policy = GeodesicFitness()
    run = run_method(world, cfg, "ga", seed=6, generations=4,
                     snapshot_generations=(3,), fitness=policy,
                     trace_operations=True, operation_trace_limit=24)
    assert len(run.operation_log) == 24
    for event in run.operation_log:
        child = evaluate(world, event.child, policy)
        assert child.fitness == event.child_score
        assert child.metrics == event.child_metrics
        assert evaluate(world, event.parent_a, policy).fitness == event.parent_a_score
        if event.parent_b is not None:
            assert evaluate(world, event.parent_b, policy).fitness == event.parent_b_score


def test_logging_is_bounded_and_absent_categories_are_safe():
    _, optimizer, *_ = _complete_one_ga_generation(trace_limit=2)
    assert len(optimizer.operation_log) == 2
    assert all(event.kind == "elite" for event in optimizer.operation_log)
    selected = select_operation_examples(optimizer.operation_log)
    assert selected.improvement is None
    assert selected.neutral is None
    assert selected.degradation is None


def test_classification_thresholds_are_explicit_and_strict_about_degradation():
    _, optimizer, *_ = _complete_one_ga_generation()
    event = next(item for item in optimizer.operation_log if item.kind == "offspring")
    tolerance = classification_tolerance(event)
    better = event.better_parent_score
    worse = min(event.parent_a_score, event.parent_b_score)
    assert classify_operation(replace(event, child_score=better + tolerance + 1e-6)) == "improvement"
    assert classify_operation(replace(event, child_score=better - tolerance / 2)) == "neutral"
    assert classify_operation(replace(event, child_score=worse - tolerance - 1e-6)) == "degradation"
    between = (better + worse) / 2
    if abs(between - better) > tolerance and between >= worse - tolerance:
        assert classify_operation(replace(event, child_score=between)) == "mixed"


def test_operator_figures_render_and_missing_category_is_labeled(tmp_path):
    synthetic_path = tmp_path / "synthetic.png"
    missing_path = tmp_path / "missing.png"
    synthetic_operation_figure(synthetic_operation(), synthetic_path)
    real_operation_figure(World(), GeodesicFitness(), None, "neutral", missing_path)
    assert synthetic_path.stat().st_size > 0
    assert missing_path.stat().st_size > 0


def test_operation_log_json_round_trip_preserves_replay_data(tmp_path):
    _, optimizer, *_ = _complete_one_ga_generation(trace_limit=8)
    path = tmp_path / "operations.json"
    save_operation_log(optimizer.operation_log, path)
    loaded = load_operation_log(path)
    assert len(loaded) == len(optimizer.operation_log)
    for expected, actual in zip(optimizer.operation_log, loaded):
        assert expected.operation_id == actual.operation_id
        assert expected.child_score == actual.child_score
        np.testing.assert_array_equal(expected.child, actual.child)
        np.testing.assert_array_equal(expected.mutation_deltas, actual.mutation_deltas)
