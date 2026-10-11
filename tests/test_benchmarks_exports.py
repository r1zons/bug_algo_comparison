import json

import numpy as np

from buglab import experiment
from buglab.animations import (
    export_comparison_gif,
    export_comparison_png,
    frame_schedule,
    trajectory_prefix,
)
from buglab.benchmark import (
    BenchmarkConfig,
    aggregate_rows,
    export_benchmark,
    export_case_study,
    run_benchmark,
    run_case_study,
    run_row,
)
from buglab.environment import World
from buglab.experiment import run_method
from buglab.optimizers import SearchConfig
from buglab.playback import ExperimentCache, PlaybackConfig


def test_first_success_uses_exact_candidate_ordinal_and_keeps_batch_end(monkeypatch):
    genomes = np.asarray([
        [1.2, 1.2, 1.2],
        [0.0, 0.0, 0.0],
        [-1.2, -1.2, -1.2],
    ])

    class ControlledSearch:
        def __init__(self, config, seed):
            self.config = config

        def ask(self):
            return genomes.copy()

        def tell(self, submitted, scores):
            np.testing.assert_array_equal(submitted, genomes)

    monkeypatch.setitem(experiment.METHODS, "controlled", ControlledSearch)
    world = World(width=12, height=10, start=(2, 5), goal=(8, 5), obstacles=(),
                  radius=.1, goal_radius=.1, step_length=2, steps=3, max_turn=1.2)
    config = SearchConfig(genome_length=3, population=3, elite=1, max_turn=1.2)
    run = run_method(world, config, "controlled", seed=0, generations=1,
                     snapshot_generations=(0,))

    assert run.first_success_evaluations == 2
    assert run.first_success_batch_end_evaluations == 3
    assert run.history[0]["first_success_eval"] == 2
    assert run.history[0]["first_success_batch_end_eval"] == 3


def test_benchmark_has_equal_budgets_and_is_deterministic():
    config = BenchmarkConfig(seeds=(2,), population=4, evaluations=8)
    first = run_benchmark(config)
    second = run_benchmark(config)
    assert {run.total_evaluations for run in first.runs} == {8}
    assert [run_row(run) for run in first.runs] == [run_row(run) for run in second.runs]
    for left, right in zip(first.runs, second.runs):
        np.testing.assert_array_equal(left.best_genome, right.best_genome)


def test_method_specific_candidate_semantics_are_not_conflated():
    cache = ExperimentCache()
    random = cache.run(PlaybackConfig(method="random", generations=2, population=4))
    local = cache.run(PlaybackConfig(method="local", generations=2, population=4))
    ga = cache.run(PlaybackConfig(method="ga", generations=2, population=4))

    assert random.candidate_semantics(1)["kind"] == "independent_samples"
    assert random.candidate_semantics(1)["independent"] is True
    local_semantics = local.candidate_semantics(1)
    assert local_semantics["kind"] == "incumbent_and_neighbors"
    assert local_semantics["incumbent_before_candidate_index"] == 0
    assert local_semantics["neighbor_candidate_indices"] == (1, 2, 3)
    assert ga.candidate_semantics(1)["kind"] == "population"
    assert ga.iteration_kind == "generation"
    assert random.iteration_kind == local.iteration_kind == "batch"


def test_playback_summary_distinguishes_candidate_roles():
    data = ExperimentCache().run(
        PlaybackConfig(method="ga", generations=2, population=4, seed=3)
    )
    summary = data.summary(1, candidate_index=0)
    assert summary["selected_candidate_index"] == 0
    assert summary["best_by_fitness_candidate_index"] == data.selected_candidate(1)
    assert summary["best_so_far_first_evaluation"] is not None
    assert "fastest_successful_candidate_index" in summary
    assert "best_so_far_goal_reached" in summary


def test_animation_schedule_holds_completed_trajectory_at_terminal_position():
    data = ExperimentCache().run(
        PlaybackConfig(method="random", generations=2, population=3, seed=1)
    )
    schedule = frame_schedule(data, episode_frames=3)
    assert schedule[0].snapshot_index == 0 and schedule[0].episode_step == 0
    assert schedule[-1].snapshot_index == 1 and schedule[-1].episode_step == data.world.steps
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    prefix = trajectory_prefix(positions, 90)
    np.testing.assert_array_equal(prefix[-1], positions[-1])


def test_rendering_cached_comparison_does_not_rerun_optimization(tmp_path):
    cache = ExperimentCache()
    datasets = {
        method: cache.run(PlaybackConfig(method=method, generations=1, population=2, seed=4))
        for method in ("ga", "random", "local")
    }
    before = cache.run_count
    png = tmp_path / "comparison.png"
    gif = tmp_path / "comparison.gif"
    export_comparison_png(datasets, png, max_candidates=2)
    export_comparison_gif(datasets, gif, episode_frames=2, fps=2, max_candidates=2)
    assert cache.run_count == before
    assert png.stat().st_size > 0 and gif.stat().st_size > 0


def test_benchmark_artifacts_and_censored_failures(tmp_path):
    config = BenchmarkConfig(map_name="dead_end", fitness_name="euclidean",
                             seeds=(91,), population=2, evaluations=2)
    result = run_benchmark(config)
    rows = [run_row(run) for run in result.runs]
    assert all(row["run_success"] == 0 for row in rows)
    assert all(row["first_success_censored"] == 1 for row in rows)
    aggregate = aggregate_rows(rows)
    assert all(row["first_success_km_median"] is None for row in aggregate)
    paths = export_benchmark(result, tmp_path)
    assert all(path.stat().st_size > 0 for path in paths.values())
    payload = json.loads(paths["json"].read_text())
    assert payload["configuration"]["evaluations"] == 2


def test_small_fitness_case_study_exports_only_outcome_tables(tmp_path):
    runs = run_case_study(maps=("spiral",), seeds=(0,), population=2, evaluations=2)
    paths = export_case_study(runs, tmp_path)
    assert all(path.stat().st_size > 0 for path in paths.values())
    header = paths["csv"].read_text().splitlines()[0]
    assert "best_fitness" not in header
    summary_header = paths["summary_csv"].read_text().splitlines()[0]
    assert "run_success_rate" in summary_header
