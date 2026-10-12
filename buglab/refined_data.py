"""Explicit source-data preparation for the focused Act I refinement scenes."""

from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .environment import World
from .experiment import run_method
from .fitness import make_fitness
from .maps import get_map, rectangular_spiral_distance_barrier
from .operator_story import save_operation_log
from .optimizers import SearchConfig
from .playback import PlaybackConfig, PlaybackData, load_playback_data, save_playback_data
from .reference_paths import rectangular_spiral_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_same_playback(expected: PlaybackData, actual: PlaybackData) -> None:
    if expected.config != actual.config or len(expected.snapshots) != len(actual.snapshots):
        raise RuntimeError("Traced GA replay does not match the reviewed playback configuration")
    for expected_snapshot, actual_snapshot in zip(expected.snapshots, actual.snapshots):
        if not (np.array_equal(expected_snapshot.candidate_genomes,
                               actual_snapshot.candidate_genomes)
                and np.array_equal(expected_snapshot.candidate_scores,
                                   actual_snapshot.candidate_scores)):
            raise RuntimeError("Operation tracing changed the reviewed GA candidate stream")


def prepare_refined_data(
    output_dir: Path,
    *,
    reviewed_dir: Path = Path("examples/act1"),
) -> dict[str, Path]:
    """Run only missing experiments and write deterministic refined-scene sources."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    paths: dict[str, Path] = {}

    # Re-run the reviewed GA with bounded tracing. Tracing consumes no random
    # values; exact snapshot equality guards that invariant before the log is used.
    reviewed_ga_path = reviewed_dir / "bridges_ga.npz"
    reviewed_ga = load_playback_data(reviewed_ga_path)
    world = World.from_map(get_map("two_bridges"))
    fitness = make_fitness("euclidean")
    search = SearchConfig(genome_length=world.steps, population=16,
                          max_turn=world.max_turn, elite=2)
    traced_run = run_method(
        world, search, "ga", 4, 14, tuple(range(14)), fitness,
        trace_operations=True, operation_trace_limit=64,
    )
    traced_playback = PlaybackData.from_run(reviewed_ga.config, world, fitness, traced_run)
    _assert_same_playback(reviewed_ga, traced_playback)
    operations_path = output_dir / "bridges_ga_operations.json"
    save_operation_log(traced_run.operation_log, operations_path)
    paths["bridges_ga_operations"] = operations_path
    records.append({
        "name": "bridges_ga_operations",
        "source_playback": str(reviewed_ga_path),
        "source_playback_sha256": _sha256(reviewed_ga_path),
        "operation_log": operations_path.name,
        "operation_log_sha256": _sha256(operations_path),
        "seed": 4,
        "evaluations": 224,
        "trace_does_not_change_candidates": True,
    })

    rectangular_world = World.from_map(get_map("rectangular_spiral"))
    rectangular_fitness = make_fitness("euclidean")
    rectangular_config = PlaybackConfig(
        map_name="rectangular_spiral", method="ga", fitness_name="euclidean",
        seed=0, generations=14, population=16,
    )
    rectangular_search = SearchConfig(
        genome_length=rectangular_world.steps,
        population=rectangular_config.population,
        max_turn=rectangular_world.max_turn,
        elite=2,
    )
    rectangular_run = run_method(
        rectangular_world, rectangular_search, "ga", rectangular_config.seed,
        rectangular_config.generations,
        tuple(range(rectangular_config.generations)), rectangular_fitness,
    )
    rectangular_playback = PlaybackData.from_run(
        rectangular_config, rectangular_world, rectangular_fitness, rectangular_run
    )
    rectangular_path = output_dir / "rectangular_spiral_ga.npz"
    save_playback_data(rectangular_playback, rectangular_path)
    paths["rectangular_spiral_ga"] = rectangular_path

    reference_world, reference_genome, reference_trace = rectangular_spiral_reference()
    if not reference_trace.reached or reference_trace.collisions:
        raise RuntimeError("Rectangular-spiral reference lost physical validity")
    certificate = rectangular_spiral_distance_barrier(reference_world.radius)
    records.append({
        "name": "rectangular_spiral_ga",
        "data_path": rectangular_path.name,
        "data_sha256": _sha256(rectangular_path),
        "config": asdict(rectangular_config),
        "evaluations": rectangular_run.total_evaluations,
        "physical_success": rectangular_run.first_success_evaluations is not None,
        "first_success_evaluation": rectangular_run.first_success_evaluations,
        "reference_genome_sha256": hashlib.sha256(reference_genome.tobytes()).hexdigest(),
        "reference_reached_goal": reference_trace.reached,
        "reference_collisions": reference_trace.collisions,
        "reference_reached_at": reference_trace.reached_at,
        "distance_barrier": asdict(certificate),
        "reference_is_optimizer_input": False,
    })

    for name in ("bridges_random", "bridges_local", "bridges_ga"):
        path = reviewed_dir / f"{name}.npz"
        records.append({"name": name, "reviewed_data_path": str(path),
                        "data_sha256": _sha256(path)})

    index_path = output_dir / "DATA_INDEX.json"
    index_path.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "Deterministic sources for Act I method and distance-barrier refinements",
        "optimization_is_explicit": True,
        "rendering_uses_only_saved_optimizer_data": True,
        "records": records,
    }, indent=2), encoding="utf-8")
    paths["index"] = index_path
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Prepare deterministic Act I refinement data")
    parser.add_argument("--output", type=Path, default=Path("examples/act1_refined"))
    parser.add_argument("--reviewed-data", type=Path, default=Path("examples/act1"))
    args = parser.parse_args(argv)
    for name, path in prepare_refined_data(args.output, reviewed_dir=args.reviewed_data).items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
