import hashlib
import json
from dataclasses import replace
from pathlib import Path

import buglab.playback as playback_module
from buglab.animations import validate_matched_playback
from buglab.playback import load_playback_data
from buglab.scenes import export_manifest, load_scene_configs


EXPECTED_KINDS = (
    "single_beetle",
    "initial_population",
    "evaluation_selection",
    "synthetic_crossover",
    "generation_transition",
    "multi_generation",
    "route_families",
    "method_comparison",
    "spiral_limitation",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_act1_manifest_declares_the_fixed_nine_scene_sequence():
    scenes = load_scene_configs(Path("act1_scenes.json"))
    assert tuple(scene.kind for scene in scenes) == EXPECTED_KINDS
    assert tuple(scene.name[:2] for scene in scenes) == tuple(f"{index:02d}" for index in range(1, 10))
    assert all(scene.output_format == "gif" for scene in scenes)
    assert all(scene.show_checkpoint_markers is False for scene in scenes)
    assert all(scene.educational_message for scene in scenes)


def test_act1_source_index_hashes_and_configs_match_saved_data():
    root = Path("examples/act1")
    index = json.loads((root / "DATA_INDEX.json").read_text())
    assert index["optimization_is_explicit"] is True
    assert index["rendering_uses_only_saved_data"] is True
    assert len(index["runs"]) == 7
    for record in index["runs"]:
        path = root / record["data_path"]
        assert _digest(path) == record["data_sha256"]
        data = load_playback_data(path)
        assert data.config.map_name == record["map_name"]
        assert data.config.method == record["method"]
        assert data.config.fitness_name == "euclidean"
        assert data.config.seed == record["seed"]
        assert data.config.generations * data.config.population == record["evaluations"]


def test_act1_comparison_sources_have_equal_budgets_and_physics():
    scene = load_scene_configs(Path("act1_scenes.json"))[7]
    assert scene.comparison_data_paths is not None
    datasets = {
        method: load_playback_data(Path(path))
        for method, path in scene.comparison_data_paths.items()
    }
    validate_matched_playback(datasets)
    assert {data.snapshots[-1].evaluations for data in datasets.values()} == {224}


def test_act1_rendering_uses_saved_data_without_optimizer_launch(monkeypatch, tmp_path):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Act I rendering must not launch an optimizer")

    monkeypatch.setattr(playback_module, "run_method", unexpected_run)
    original = load_scene_configs(Path("act1_scenes.json"))[5]
    short = replace(original, name="short_scene", duration_seconds=.2, fps=5)
    [(movie, metadata)] = export_manifest((short,), tmp_path)
    assert movie.stat().st_size > 0
    payload = json.loads(metadata.read_text())
    assert payload["rendering_launches_optimizer"] is False
    assert payload["source_data_paths"]["primary"] == original.data_path
    assert payload["export"]["canvas_pixels"] == [1280, 720]


def test_act1_exports_have_complete_provenance_and_honest_spiral_result():
    folder = Path("outputs/act1")
    scenes = load_scene_configs(Path("act1_scenes.json"))
    for scene in scenes:
        assert (folder / f"{scene.name}.gif").stat().st_size > 0
        payload = json.loads((folder / f"{scene.name}.json").read_text())
        assert payload["scene"]["name"] == scene.name
        assert payload["export"]["canvas_pixels"] == [1280, 720]
        assert payload["rendering_launches_optimizer"] is False
    spiral = json.loads((folder / "09_spiral_fitness_problem.json").read_text())
    assert spiral["source_playback"]["fitness_name"] == "euclidean"
    assert spiral["source_details"]["run_goal_reached"] is False
    assert spiral["source_details"]["corridor_has_distance_increase"] is True
