import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

import buglab.playback as playback_module
from buglab.playback import ExperimentCache, PlaybackConfig, trajectory_figure
from buglab.scenes import (
    LectureStyle,
    SceneConfig,
    available_export_formats,
    export_scene,
    load_scene_configs,
)


def _recorded_data():
    cache = ExperimentCache()
    data = cache.run(PlaybackConfig(
        map_name="two_bridges", method="random", fitness_name="euclidean",
        seed=7, generations=1, population=3,
    ))
    return cache, data


def test_scene_manifest_is_deterministic_and_uses_fixed_16_by_9_style():
    first = load_scene_configs(Path("pilot_scenes.json"))
    second = load_scene_configs(Path("pilot_scenes.json"))
    assert first == second
    assert tuple(scene.kind for scene in first) == (
        "single_beetle", "initial_population", "synthetic_crossover"
    )
    assert all(scene.show_checkpoint_markers is False for scene in first)
    style = LectureStyle()
    assert np.isclose(style.canvas_width / style.canvas_height, 16 / 9)
    assert "gif" in available_export_formats()


def test_checkpoint_marker_toggle_is_display_only(monkeypatch):
    cache, data = _recorded_data()
    before_scores = data.snapshot(0).candidate_scores.copy()
    for candidate in range(data.config.population):
        data.candidate_rollout(0, candidate)
    data.best_rollout(0)
    before_runs = cache.run_count

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("display toggle must not calculate a reward or replay physics")

    monkeypatch.setattr(playback_module, "evaluate", unexpected_evaluation)
    hidden = trajectory_figure(data, 0, show_checkpoints=False)
    visible = trajectory_figure(data, 0, show_checkpoints=True)
    assert "Ordered checkpoints" not in [item.get_label() for item in hidden.axes[0].collections]
    assert "Ordered checkpoints" in [item.get_label() for item in visible.axes[0].collections]
    plt.close(hidden); plt.close(visible)
    np.testing.assert_array_equal(data.snapshot(0).candidate_scores, before_scores)
    assert cache.run_count == before_runs


def test_renderers_use_recorded_data_without_running_optimizer(tmp_path):
    cache, data = _recorded_data()
    before_runs = cache.run_count
    config = SceneConfig(
        name="single", kind="single_beetle", title="Recorded beetle",
        duration_seconds=.2, fps=5,
    )
    movie, metadata = export_scene(config, tmp_path, data=data)
    assert cache.run_count == before_runs
    assert movie.stat().st_size > 0 and metadata.stat().st_size > 0
    payload = json.loads(metadata.read_text())
    assert payload["source_details"]["illustrative"] is False
    assert payload["source_playback"]["fitness_name"] == "euclidean"
    assert payload["rendering_launches_optimizer"] is False


def test_synthetic_crossover_export_is_explicitly_illustrative(tmp_path):
    config = SceneConfig(
        name="crossover", kind="synthetic_crossover", title="Synthetic crossover",
        duration_seconds=.2, fps=5,
    )
    movie, metadata = export_scene(config, tmp_path)
    assert movie.stat().st_size > 0
    payload = json.loads(metadata.read_text())
    assert payload["source_details"]["illustrative"] is True
    assert payload["source_details"]["cut"] == 4
    assert payload["source_details"]["mutated_indices"] == [2, 6]
