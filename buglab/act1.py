"""Explicit, reproducible data preparation for the fixed Act I lecture package."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .environment import World
from .experiment import run_method
from .fitness import make_fitness
from .maps import get_map
from .operator_story import save_operation_log
from .optimizers import SearchConfig
from .playback import PlaybackConfig, PlaybackData, save_playback_data


@dataclass(frozen=True)
class Act1RunSpec:
    """One intentional source run used by one or more lecture scenes."""

    name: str
    map_name: str
    method: str
    fitness_name: str
    seed: int
    generations: int
    population: int
    trace_first_transition: bool = False


ACT1_RUNS = (
    Act1RunSpec("open_random", "open_field", "random", "euclidean", 12, 1, 12),
    Act1RunSpec("open_ga", "open_field", "ga", "euclidean", 12, 2, 12, True),
    Act1RunSpec("simple_ga", "default", "ga", "euclidean", 0, 12, 16, True),
    Act1RunSpec("bridges_ga", "two_bridges", "ga", "euclidean", 4, 14, 16),
    Act1RunSpec("bridges_random", "two_bridges", "random", "euclidean", 4, 14, 16),
    Act1RunSpec("bridges_local", "two_bridges", "local", "euclidean", 4, 14, 16),
    Act1RunSpec("spiral_ga", "spiral", "ga", "euclidean", 0, 10, 16),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_act1_data(folder: Path) -> dict[str, Path]:
    """Run the declared experiments explicitly and save playback/operation records."""
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    records = []
    for spec in ACT1_RUNS:
        world = World.from_map(get_map(spec.map_name))
        fitness = make_fitness(spec.fitness_name)
        search = SearchConfig(
            genome_length=world.steps,
            population=spec.population,
            max_turn=world.max_turn,
            elite=min(2, spec.population),
        )
        run = run_method(
            world, search, spec.method, spec.seed, spec.generations,
            tuple(range(spec.generations)), fitness,
            trace_operations=spec.trace_first_transition,
            operation_trace_limit=spec.population if spec.trace_first_transition else 0,
        )
        config = PlaybackConfig(
            map_name=spec.map_name,
            method=spec.method,
            fitness_name=spec.fitness_name,
            seed=spec.seed,
            generations=spec.generations,
            population=spec.population,
        )
        playback = PlaybackData.from_run(config, world, fitness, run)
        data_path = folder / f"{spec.name}.npz"
        save_playback_data(playback, data_path)
        paths[spec.name] = data_path
        record = {
            **asdict(spec),
            "data_path": data_path.name,
            "data_sha256": _sha256(data_path),
            "evaluations": spec.generations * spec.population,
            "physical_success": run.first_success_evaluations is not None,
            "first_success_evaluation": run.first_success_evaluations,
        }
        if spec.trace_first_transition:
            operations_path = folder / f"{spec.name}_operations.json"
            save_operation_log(run.operation_log, operations_path)
            paths[f"{spec.name}_operations"] = operations_path
            record["operation_log_path"] = operations_path.name
            record["operation_log_sha256"] = _sha256(operations_path)
            record["recorded_operations"] = len(run.operation_log)
        records.append(record)
    index = folder / "DATA_INDEX.json"
    index.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "Deterministic source data for the fixed Act I lecture animations",
        "optimization_is_explicit": True,
        "rendering_uses_only_saved_data": True,
        "runs": records,
    }, indent=2), encoding="utf-8")
    paths["index"] = index
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Prepare deterministic Act I playback data")
    parser.add_argument("--output", type=Path, default=Path("examples/act1"))
    args = parser.parse_args(argv)
    for name, path in prepare_act1_data(args.output).items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
