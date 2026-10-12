import hashlib
import json
from pathlib import Path

import numpy as np

from buglab.crossover_slides import (
    CHILD_BEFORE_DEGREES,
    CUT_INDEX,
    FINAL_CHILD_DEGREES,
    MUTATED_INDICES,
    PARENT_A_DEGREES,
    PARENT_B_DEGREES,
    crossover_genomes_radians,
    crossover_world,
    export_crossover_slides,
)
from buglab.environment import EpisodeTrace, World, simulate
from buglab.fitness import EuclideanFitness, FinalEuclideanFitness, make_fitness
from buglab.playback import load_playback_data
from buglab.scenes import (
    _barrier_crossing_y,
    _corridor_reference_trace,
    _route_family,
    interpolated_trajectory_prefix,
    load_scene_configs,
)


def test_final_euclidean_is_explicit_and_legacy_euclidean_remains_closest_approach():
    world = World(width=20, height=20, start=(2, 10), goal=(10, 10), obstacles=(), steps=2)
    trace = EpisodeTrace(
        positions=np.asarray(((2, 10), (8, 10), (5, 10)), dtype=float),
        reached=False, reached_at=None, collisions=0, turns=np.zeros(2),
    )
    assert EuclideanFitness().score(world, trace).fitness == -2.0
    assert FinalEuclideanFitness().score(world, trace).fitness == -5.0
    assert make_fitness("euclidean").name == "euclidean"
    assert make_fitness("final_euclidean").name == "final_euclidean"


def test_interpolated_prefix_preserves_recorded_vertices_and_interpolates_only_between_them():
    positions = np.asarray(((0, 0), (2, 0), (2, 2)), dtype=float)
    np.testing.assert_array_equal(interpolated_trajectory_prefix(positions, 1), positions[:2])
    np.testing.assert_allclose(interpolated_trajectory_prefix(positions, 1.5),
                               ((0, 0), (2, 0), (2, 1)))


def test_two_bridges_classification_requires_a_real_centreline_crossing():
    lower = np.asarray(((40, 23), (47, 23), (53, 23), (60, 23)), dtype=float)
    upper = np.asarray(((40, 57), (60, 57)), dtype=float)
    nearby_only = np.asarray(((43, 23), (49.9, 23), (43, 23)), dtype=float)
    assert _barrier_crossing_y(lower) == 23.0
    assert _route_family(lower) == "lower crossing"
    assert _route_family(upper) == "upper crossing"
    assert _route_family(nearby_only) == "did not cross"


def test_spiral_reference_is_physics_valid_and_contains_an_increasing_detour_segment():
    data = load_playback_data(Path("examples/act1/spiral_ga.npz"))
    trace = _corridor_reference_trace(data)
    distances = np.linalg.norm(trace.positions - np.asarray(data.world.goal), axis=1)
    increases = np.flatnonzero(np.diff(distances) > 1e-6)
    assert trace.reached
    assert trace.collisions == 0
    assert len(increases) > 0
    assert np.max(np.abs(trace.turns)) <= data.world.max_turn + 1e-12


def test_static_crossover_reconstruction_and_simulated_trajectories_are_exact():
    genomes = crossover_genomes_radians()
    np.testing.assert_array_equal(
        CHILD_BEFORE_DEGREES,
        np.concatenate((PARENT_A_DEGREES[:CUT_INDEX], PARENT_B_DEGREES[CUT_INDEX:])),
    )
    changed = np.flatnonzero(FINAL_CHILD_DEGREES != CHILD_BEFORE_DEGREES)
    np.testing.assert_array_equal(changed, MUTATED_INDICES)
    world = crossover_world()
    for genome in genomes.values():
        assert genome.shape == (world.steps,)
        assert np.all(np.abs(genome) <= world.max_turn)
        assert len(simulate(world, genome).positions) >= 2


def test_final_manifest_replaces_main_scene_04_with_static_sequence():
    payload = json.loads(Path("act1_final_scenes.json").read_text())
    scenes = load_scene_configs(Path("act1_final_scenes.json"))
    assert tuple(scene.name[:2] for scene in scenes) == ("01", "02", "03", "05", "06", "07", "08", "09")
    assert all(scene.fps >= 12 for scene in scenes)
    assert payload["static_sequence"] == [
        "04a_crossover_parents", "04b_crossover_child",
        "04c_crossover_mutation", "04d_crossover_outcome",
    ]


def test_static_crossover_exports_include_hashes_and_do_not_launch_optimizer(tmp_path):
    exports = export_crossover_slides(tmp_path / "first")
    repeated = export_crossover_slides(tmp_path / "repeated")
    assert len(exports) == 4
    for (png, svg, sidecar), (repeated_png, _, _) in zip(exports, repeated):
        payload = json.loads(sidecar.read_text())
        assert payload["rendering_launches_optimizer"] is False
        assert payload["render"]["png_sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()
        assert payload["render"]["svg_sha256"] == hashlib.sha256(svg.read_bytes()).hexdigest()
        assert png.read_bytes() == repeated_png.read_bytes()


def test_final_gif_sidecars_match_exported_artifact_hashes():
    output = Path("outputs/act1_final")
    for scene in load_scene_configs(Path("act1_final_scenes.json")):
        gif = output / f"{scene.name}.gif"
        sidecar = json.loads((output / f"{scene.name}.json").read_text())
        assert sidecar["export"]["artifact_sha256"] == hashlib.sha256(gif.read_bytes()).hexdigest()
        assert sidecar["rendering_launches_optimizer"] is False
