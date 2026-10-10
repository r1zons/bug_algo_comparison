import json
from pathlib import Path

import numpy as np

from buglab.notebook_ui import PlaybackWidget, create_playback_widget
from buglab.playback import (
    ExperimentCache,
    PlaybackConfig,
    export_generation_png,
    export_summary_png,
    load_playback_data,
    save_playback_data,
)


def test_cached_playback_round_trip_is_deterministic(tmp_path):
    cache = ExperimentCache()
    config = PlaybackConfig(method="ga", seed=4, generations=3, population=5)
    original = cache.run(config)
    path = tmp_path / "run.npz"
    save_playback_data(original, path)
    loaded = load_playback_data(path)

    assert loaded.config == original.config
    assert loaded.generations == (0, 1, 2)
    assert loaded.history == original.history
    for before, after in zip(original.snapshots, loaded.snapshots):
        np.testing.assert_array_equal(before.candidate_genomes, after.candidate_genomes)
        np.testing.assert_array_equal(before.candidate_scores, after.candidate_scores)
        np.testing.assert_array_equal(before.best_genome, after.best_genome)
    first = loaded.candidate_rollout(1, 2)
    second = loaded.candidate_rollout(1, 2)
    assert first is second
    np.testing.assert_array_equal(first.positions, second.positions)


def test_generation_selection_maps_to_recorded_snapshot_metadata():
    data = ExperimentCache().run(
        PlaybackConfig(method="random", seed=2, generations=4, population=6)
    )
    for index, generation in enumerate(data.generations):
        assert data.snapshot_index(generation) == index
        assert data.snapshot_for_generation(generation) is data.snapshot(index)
        assert data.summary(index)["generation"] == generation
        assert data.summary(index)["evaluations"] == (generation + 1) * 6


def test_playback_supports_all_methods_with_same_data_contract():
    cache = ExperimentCache()
    for method in ("ga", "random", "local"):
        data = cache.run(PlaybackConfig(method=method, generations=2, population=4))
        assert len(data.snapshots) == 2
        assert data.snapshot(1).candidate_genomes.shape == (4, 90)
        assert data.method_description
        assert data.summary(1)["method"] == method


def test_no_success_run_has_graceful_summary():
    data = ExperimentCache().run(
        PlaybackConfig(map_name="dead_end", method="random", fitness_name="euclidean",
                       seed=91, generations=1, population=2)
    )
    assert data.history[-1]["goal_reached"] == 0
    summary = data.summary(0)
    assert summary["goal_reached"] is False
    assert summary["steps_to_goal"] is None
    assert summary["evaluations"] == 2


def test_playback_controls_do_not_rerun_cached_experiment():
    cache = ExperimentCache()
    config = PlaybackConfig(generations=2, population=4)
    data = cache.run(config)
    assert cache.run_count == 1
    assert cache.run(config) is data
    data.candidate_rollout(0, 0)
    data.best_rollout(1)
    data.summary(1)
    assert cache.run_count == 1


def test_widget_generation_slider_uses_snapshot_generation_without_rerun():
    cache = ExperimentCache()
    data = cache.run(PlaybackConfig(method="local", generations=3, population=4))
    ui = create_playback_widget(cache)
    assert isinstance(ui, PlaybackWidget)
    assert ui.current is data
    assert (ui.generation.min, ui.generation.max) == (0, 2)
    before = cache.run_count
    ui.generation.value = 2
    assert ui.current.snapshot_index(ui.generation.value) == 2
    assert cache.run_count == before


def test_generation_and_summary_exports_produce_png_files(tmp_path):
    data = ExperimentCache().run(
        PlaybackConfig(method="ga", seed=1, generations=2, population=4)
    )
    generation_path = tmp_path / "generation.png"
    summary_path = tmp_path / "summary.png"
    export_generation_png(data, 1, generation_path, show_collisions=True, top_k=3)
    export_summary_png(data, 1, summary_path)
    assert generation_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0


def test_notebook_is_thin_and_contains_act_i_demo():
    notebook = json.loads(Path("buglab_demo.ipynb").read_text())
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    assert any("Act I Demo" in source for source in sources)
    code = "\n".join(
        source for source, cell in zip(sources, notebook["cells"])
        if cell["cell_type"] == "code"
    )
    assert "run_method" not in code
    assert "GeneticAlgorithm" not in code


def test_bundled_demo_cache_loads_with_expected_metadata():
    data = load_playback_data(Path("examples/task4_demo.npz"))
    assert data.config == PlaybackConfig(
        map_name="default", method="ga", fitness_name="geodesic",
        seed=0, generations=8, population=16,
    )
    assert data.generations == tuple(range(8))
    assert data.summary(7)["evaluations"] == 128
