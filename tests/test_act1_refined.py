import hashlib
import json
from pathlib import Path

import numpy as np

import buglab.experiment as experiment_module
from buglab.animations import validate_matched_playback
from buglab.environment import EpisodeTrace, World
from buglab.fitness import EuclideanFitness, FinalEuclideanFitness
from buglab.maps import MAPS, RECTANGULAR_SPIRAL, rectangular_spiral_distance_barrier
from buglab.optimizers import LocalSearch, RandomSearch, SearchConfig
from buglab.operator_story import load_operation_log
from buglab.playback import load_playback_data
from buglab.reference_paths import rectangular_spiral_reference
from buglab.refined_scenes import export_refined_manifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rectangular_spiral_has_an_analytic_positive_distance_barrier():
    certificate = rectangular_spiral_distance_barrier(radius=1.1)
    lower_wall = RECTANGULAR_SPIRAL.obstacles[0]
    assert lower_wall.xmin == 0.0
    assert RECTANGULAR_SPIRAL.start[1] < lower_wall.ymin - 1.1
    assert RECTANGULAR_SPIRAL.goal[1] > lower_wall.ymax + 1.1
    assert certificate.separator_x == lower_wall.xmax + 1.1
    assert certificate.required_distance > certificate.starting_distance
    assert certificate.delta > 30.0


def test_rectangular_spiral_reference_reaches_with_unchanged_real_physics():
    world, genome, trace = rectangular_spiral_reference()
    assert world.steps == 90
    assert world.step_length == 1.55
    assert world.max_turn == 0.42
    assert genome.shape == (world.steps,)
    assert np.all(np.abs(genome) <= world.max_turn + 1e-12)
    assert trace.reached and trace.reached_at == 76
    assert trace.collisions == 0


def test_closest_approach_plateaus_while_final_distance_worsens_on_detour():
    world, _, trace = rectangular_spiral_reference()
    distances = np.linalg.norm(trace.positions - np.asarray(world.goal), axis=1)
    maximum = int(np.argmax(distances))
    partial = EpisodeTrace(trace.positions[:maximum + 1], False, None, 0,
                           trace.turns[:maximum])
    closest = EuclideanFitness().score(world, partial)
    final = FinalEuclideanFitness().score(world, partial)
    assert closest.fitness == -float(distances[:maximum + 1].min())
    assert final.fitness == -float(distances[maximum])
    assert closest.fitness > final.fitness
    running_minimum = np.minimum.accumulate(distances[:maximum + 1])
    assert np.any((np.diff(distances[:maximum + 1]) > 0)
                  & np.isclose(np.diff(running_minimum), 0))


def test_random_search_batches_are_fresh_and_tell_does_not_create_lineage():
    config = SearchConfig(genome_length=12, population=5)
    first = RandomSearch(config, seed=9)
    control = RandomSearch(config, seed=9)
    first_batch = first.ask()
    control_first = control.ask()
    np.testing.assert_array_equal(first_batch, control_first)
    first.tell(first_batch, np.arange(config.population, dtype=float))
    second_batch = first.ask()
    control_second = control.ask()
    np.testing.assert_array_equal(second_batch, control_second)
    assert not np.array_equal(first_batch, second_batch)


def test_local_search_keeps_incumbent_and_accepts_only_between_episode_batches():
    config = SearchConfig(genome_length=10, population=4, mutation_rate=1.0)
    search = LocalSearch(config, seed=3)
    initial = search.ask()
    initial_scores = np.asarray((0.0, 1.0, 4.0, 2.0))
    search.tell(initial, initial_scores)
    neighborhood = search.ask()
    np.testing.assert_array_equal(neighborhood[0], initial[2])
    assert all(not np.array_equal(row, neighborhood[0]) for row in neighborhood[1:])

    improving_scores = np.asarray((4.0, 3.0, 7.0, 2.0))
    search.tell(neighborhood, improving_scores)
    improved = neighborhood[2].copy()
    next_neighborhood = search.ask()
    np.testing.assert_array_equal(next_neighborhood[0], improved)

    rejected_scores = np.asarray((7.0, 6.0, 5.0, 4.0))
    search.tell(next_neighborhood, rejected_scores)
    after_rejection = search.ask()
    np.testing.assert_array_equal(after_rejection[0], improved)


def test_recorded_local_semantics_identify_best_neighbor_and_acceptance():
    data = load_playback_data(Path("examples/act1/bridges_local.npz"))
    rejected = data.candidate_semantics(4)
    accepted = data.candidate_semantics(6)
    for snapshot_index, semantics in ((4, rejected), (6, accepted)):
        snapshot = data.snapshot(snapshot_index)
        neighbor = semantics["best_neighbor_candidate_index"]
        assert neighbor != 0
        assert semantics["best_neighbor_score"] == snapshot.candidate_scores[neighbor]
        assert neighbor == int(np.argmax(snapshot.candidate_scores[1:])) + 1
    assert rejected["accepted"] is False
    assert accepted["accepted"] is True


def test_matched_comparison_has_equal_budget_seed_physics_and_fitness():
    datasets = {
        method: load_playback_data(Path(f"examples/act1/bridges_{method}.npz"))
        for method in ("ga", "random", "local")
    }
    validate_matched_playback(datasets)
    assert {data.config.seed for data in datasets.values()} == {4}
    assert {data.config.fitness_name for data in datasets.values()} == {"euclidean"}
    assert {data.snapshots[-1].evaluations for data in datasets.values()} == {224}


def test_refined_render_is_deterministic_aligned_and_never_runs_optimizer(monkeypatch, tmp_path):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Rendering must not launch optimization")

    monkeypatch.setattr(experiment_module, "run_method", unexpected_run)
    source = Path("examples/act1/bridges_random.npz").resolve()
    manifest = tmp_path / "short.json"
    manifest.write_text(json.dumps({"scenes": [{
        "name": "short_random", "kind": "random_mechanics",
        "duration_seconds": .35, "fps": 12,
        "educational_message": "fresh sampled candidate",
        "data_path": str(source), "snapshot_indices": [0], "max_candidates": 3,
    }]}))
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    export_refined_manifest(manifest, first_output)
    export_refined_manifest(manifest, second_output)
    first_gif = first_output / "short_random.gif"
    assert first_gif.read_bytes() == (second_output / "short_random.gif").read_bytes()
    metadata = json.loads((first_output / "short_random.json").read_text())
    assert metadata["rendering_launches_optimizer"] is False
    assert metadata["export"]["gif_sha256"] == _digest(first_gif)
    record = metadata["source_details"]["candidate_records"][0]
    data = load_playback_data(source)
    snapshot = data.snapshot(record["snapshot_index"])
    candidate = record["candidate_index"]
    assert record["score"] == snapshot.candidate_scores[candidate]
    assert record["genome_sha256"] == hashlib.sha256(
        snapshot.candidate_genomes[candidate].astype(np.float64).tobytes()
    ).hexdigest()


def test_refined_recorded_optimizer_source_is_honest_and_reproducible():
    index = json.loads(Path("examples/act1_refined/DATA_INDEX.json").read_text())
    record = next(row for row in index["records"] if row["name"] == "rectangular_spiral_ga")
    path = Path("examples/act1_refined") / record["data_path"]
    assert _digest(path) == record["data_sha256"]
    assert record["physical_success"] is False
    assert record["reference_reached_goal"] is True
    assert record["reference_is_optimizer_input"] is False
    data = load_playback_data(path)
    assert data.config.map_name == "rectangular_spiral"
    assert data.config.fitness_name == "euclidean"
    assert data.snapshots[-1].evaluations == 224


def test_recorded_ga_operation_matches_the_displayed_evaluated_candidate():
    data = load_playback_data(Path("examples/act1/bridges_ga.npz"))
    operations = load_operation_log(
        Path("examples/act1_refined/bridges_ga_operations.json")
    )
    event = next(operation for operation in operations
                 if operation.kind == "offspring" and operation.generation == 1
                 and operation.mutated_indices)
    snapshot = data.snapshot(data.snapshot_index(event.generation))
    np.testing.assert_array_equal(
        event.child, snapshot.candidate_genomes[event.candidate_index]
    )
    assert event.child_score == snapshot.candidate_scores[event.candidate_index]
    reconstructed = np.concatenate((
        event.parent_a[:event.cut], event.parent_b[event.cut:]
    ))
    np.testing.assert_array_equal(reconstructed, event.child_before_mutation)
