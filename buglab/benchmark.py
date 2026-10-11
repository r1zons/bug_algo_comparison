"""Reproducible multi-seed benchmarks and headless Task 5 exports."""

from argparse import ArgumentParser
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

from matplotlib import pyplot as plt
import numpy as np

from .animations import export_comparison_gif, export_comparison_png, export_method_gif
from .environment import World
from .experiment import Run, run_method
from .fitness import FITNESS_NAMES, make_fitness
from .maps import MAPS, get_map
from .optimizers import METHODS, SearchConfig
from .playback import PlaybackConfig, PlaybackData


@dataclass(frozen=True)
class BenchmarkConfig:
    """A matched optimizer benchmark with an exact candidate-evaluation budget."""

    map_name: str = "default"
    fitness_name: str = "geodesic"
    methods: tuple[str, ...] = ("ga", "random", "local")
    seeds: tuple[int, ...] = (0, 1, 2)
    population: int = 16
    evaluations: int = 128
    collision_penalty: float = 0.5
    smoothness_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.map_name not in MAPS:
            raise ValueError(f"Unknown map {self.map_name!r}")
        if self.fitness_name not in FITNESS_NAMES:
            raise ValueError(f"Unknown fitness {self.fitness_name!r}")
        if not self.methods or any(method not in METHODS for method in self.methods):
            raise ValueError("methods must contain registered optimizer names")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.population <= 0 or self.evaluations <= 0:
            raise ValueError("population and evaluations must be positive")
        if self.evaluations % self.population:
            raise ValueError("evaluation budget must be divisible by population")

    @property
    def batches(self) -> int:
        return self.evaluations // self.population

    @classmethod
    def smoke(cls) -> "BenchmarkConfig":
        return cls()

    @classmethod
    def full(cls) -> "BenchmarkConfig":
        """A configurable lecture benchmark; intentionally never the default."""
        return cls(seeds=tuple(range(25)), population=64, evaluations=4160)


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    runs: list[Run]
    elapsed_seconds: float


def run_benchmark(config: BenchmarkConfig, *, record_snapshots: bool = False) -> BenchmarkResult:
    """Execute matched runs. Every candidate, including re-evaluated elites, counts once."""
    world = World.from_map(get_map(config.map_name))
    fitness = make_fitness(
        config.fitness_name,
        collision_penalty=config.collision_penalty,
        smoothness_weight=config.smoothness_weight,
    )
    search = SearchConfig(
        genome_length=world.steps,
        population=config.population,
        max_turn=world.max_turn,
        elite=min(2, config.population),
    )
    snapshot_indices = tuple(range(config.batches)) if record_snapshots else (config.batches - 1,)
    started = time.perf_counter()
    runs = [
        run_method(
            world, search, method, seed, config.batches, snapshot_indices, fitness
        )
        for method in config.methods
        for seed in config.seeds
    ]
    return BenchmarkResult(config, runs, time.perf_counter() - started)


def _outcome(run: Run):
    return run.best_success_rollout or run.best_rollout


def run_row(run: Run) -> dict[str, Any]:
    """Flatten one run while keeping run success separate from genome frequency."""
    outcome = _outcome(run)
    assert outcome.metrics is not None
    return {
        "method": run.method,
        "seed": run.seed,
        "map": run.map_name,
        "fitness": run.fitness_name,
        "population": run.search_config["population"],
        "genome_length": run.search_config["genome_length"],
        "max_turn": run.search_config["max_turn"],
        "init_sigma": run.search_config["init_sigma"],
        "mutation_sigma": run.search_config["mutation_sigma"],
        "mutation_rate": run.search_config["mutation_rate"],
        "tournament": run.search_config["tournament"],
        "elite": run.search_config["elite"],
        "field_width": run.physics_config["width"],
        "field_height": run.physics_config["height"],
        "beetle_radius": run.physics_config["radius"],
        "goal_radius": run.physics_config["goal_radius"],
        "step_length": run.physics_config["step_length"],
        "episode_steps": run.physics_config["episode_steps"],
        "grid_resolution": run.physics_config["grid_resolution"],
        "evaluations": run.total_evaluations,
        "batches": run.total_evaluations // run.search_config["population"],
        "run_success": int(run.first_success_evaluations is not None),
        "first_success_evaluations": run.first_success_evaluations,
        "first_success_batch_end_evaluations": run.first_success_batch_end_evaluations,
        "first_success_censored": int(run.first_success_evaluations is None),
        "first_success_observed_or_censor": (
            run.first_success_evaluations
            if run.first_success_evaluations is not None
            else run.total_evaluations
        ),
        "evaluated_successful_genomes": run.successful_evaluations,
        "evaluated_genome_success_fraction": run.successful_evaluations / run.total_evaluations,
        "best_fitness": run.best_rollout.fitness,
        "best_by_fitness_goal_reached": int(run.best_rollout.reached),
        "representative_outcome": "fastest_success" if run.best_success_rollout else "best_by_fitness_failure",
        "fastest_success_evaluation": run.best_success_evaluation,
        "steps_to_goal": outcome.metrics.steps_to_goal,
        "collisions": outcome.metrics.collision_count,
        "path_length": outcome.metrics.actual_path_length,
        "final_distance": outcome.metrics.final_distance,
        "environment_steps": run.environment_steps,
    }


def _median_iqr(values: Iterable[float | int | None]) -> tuple[float | None, float | None, float | None]:
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(clean):
        return None, None, None
    return tuple(float(value) for value in np.percentile(clean, (50, 25, 75)))


def kaplan_meier_median(rows: list[dict[str, Any]]) -> float | None:
    """Median evaluations to success with unsuccessful runs right-censored at budget."""
    observations = sorted(
        (int(row["first_success_observed_or_censor"]), not bool(row["first_success_censored"]))
        for row in rows
    )
    survival = 1.0
    at_risk = len(observations)
    for time_value in sorted({time_value for time_value, _ in observations}):
        events = sum(event for time_at, event in observations if time_at == time_value)
        censored = sum(not event for time_at, event in observations if time_at == time_value)
        if at_risk and events:
            survival *= 1.0 - events / at_risk
            if survival <= .5:
                return float(time_value)
        at_risk -= events + censored
    return None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["map"], row["fitness"], row["method"]), []).append(row)
    aggregates = []
    for (map_name, fitness_name, method), group in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "map": map_name,
            "fitness": fitness_name,
            "method": method,
            "runs": len(group),
            "successful_runs": sum(row["run_success"] for row in group),
            "run_success_rate": sum(row["run_success"] for row in group) / len(group),
            "censored_runs": sum(row["first_success_censored"] for row in group),
            "first_success_km_median": kaplan_meier_median(group),
            "evaluated_genome_success_fraction": float(np.mean([
                row["evaluated_genome_success_fraction"] for row in group
            ])),
            "evaluations_per_run": group[0]["evaluations"],
        }
        for key in (
            "best_fitness", "first_success_evaluations", "final_distance",
            "steps_to_goal", "collisions", "path_length", "environment_steps",
        ):
            median, q1, q3 = _median_iqr(row[key] for row in group)
            aggregate[f"{key}_median"] = median
            aggregate[f"{key}_q1"] = q1
            aggregate[f"{key}_q3"] = q3
        aggregates.append(aggregate)
    return aggregates


def progress_rows(runs: list[Run]) -> list[dict[str, Any]]:
    """Median/IQR best-so-far fitness; valid only within one fitness policy."""
    groups: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.map_name, run.fitness_name, run.method), []).append(run)
    rows = []
    for (map_name, fitness_name, method), group in sorted(groups.items()):
        for history_index, reference in enumerate(group[0].history):
            values = [run.history[history_index]["best_fitness"] for run in group]
            median, q1, q3 = _median_iqr(values)
            rows.append({
                "map": map_name, "fitness": fitness_name, "method": method,
                "evaluations": reference["evaluations"],
                "best_fitness_median": median, "best_fitness_q1": q1,
                "best_fitness_q3": q3, "seeds": len(group),
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def progress_figure(result: BenchmarkResult, path: Path) -> None:
    """Plot multi-seed median and IQR against equal evaluation budgets."""
    rows = progress_rows(result.runs)
    colors = {"ga": "#168579", "random": "#d17a22", "local": "#3973ac"}
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="white")
    for method in result.config.methods:
        selected = [row for row in rows if row["method"] == method]
        x = np.asarray([row["evaluations"] for row in selected])
        median = np.asarray([row["best_fitness_median"] for row in selected])
        q1 = np.asarray([row["best_fitness_q1"] for row in selected])
        q3 = np.asarray([row["best_fitness_q3"] for row in selected])
        ax.plot(x, median, lw=2.5, color=colors[method], label=method.upper())
        ax.fill_between(x, q1, q3, color=colors[method], alpha=.16)
    ax.set_xlabel("fitness evaluations", fontsize=12)
    ax.set_ylabel("best-so-far fitness (median and IQR)", fontsize=12)
    ax.set_title(
        f"Matched benchmark · map={result.config.map_name} · "
        f"fitness={result.config.fitness_name} · {len(result.config.seeds)} seeds\n"
        "Raw fitness comparison is valid here because every method uses the same policy",
        fontsize=15,
    )
    ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white"); plt.close(fig)


def export_benchmark(result: BenchmarkResult, folder: Path) -> dict[str, Path]:
    """Write reproducible per-run, aggregate, progress, JSON, and interpretation outputs."""
    folder.mkdir(parents=True, exist_ok=True)
    per_run = [
        {
            **run_row(run),
            "collision_penalty_parameter": result.config.collision_penalty,
            "smoothness_weight_parameter": result.config.smoothness_weight,
        }
        for run in result.runs
    ]
    aggregate = aggregate_rows(per_run)
    progress = progress_rows(result.runs)
    paths = {
        "per_run_csv": folder / "benchmark_runs.csv",
        "aggregate_csv": folder / "benchmark_summary.csv",
        "progress_csv": folder / "benchmark_progress.csv",
        "json": folder / "benchmark.json",
        "progress_png": folder / "benchmark_progress.png",
        "interpretation": folder / "INTERPRETATION.md",
    }
    _write_csv(paths["per_run_csv"], per_run)
    _write_csv(paths["aggregate_csv"], aggregate)
    _write_csv(paths["progress_csv"], progress)
    payload = {
        "schema_version": 1,
        "configuration": asdict(result.config),
        "evaluation_policy": (
            "Every submitted candidate counts, including re-evaluated GA elites and the "
            "re-evaluated local-search incumbent. Cached rendering adds no optimizer evaluations."
        ),
        "elapsed_seconds": result.elapsed_seconds,
        "runs": per_run,
        "aggregate": aggregate,
        "progress": progress,
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    progress_figure(result, paths["progress_png"])
    _write_interpretation(result, aggregate, paths["interpretation"])
    return paths


def _write_interpretation(
    result: BenchmarkResult, aggregate: list[dict[str, Any]], path: Path
) -> None:
    lines = [
        "# Benchmark interpretation",
        "",
        f"This {'smoke demonstration' if len(result.config.seeds) < 20 else 'multi-seed benchmark'} "
        f"uses map `{result.config.map_name}`, fitness `{result.config.fitness_name}`, "
        f"{len(result.config.seeds)} seeds, and {result.config.evaluations} candidate evaluations per run.",
        "",
        "Run-level success means at least one evaluated genome physically reached the goal. "
        "Evaluated-genome success fraction is a different descriptive quantity and can include "
        "repeated elites or incumbents; it is not a count of independent successes.",
        "",
    ]
    for row in aggregate:
        km = "not reached" if row["first_success_km_median"] is None else str(row["first_success_km_median"])
        lines.append(
            f"- {row['method']}: {row['successful_runs']}/{row['runs']} successful runs; "
            f"Kaplan–Meier median evaluations to first success: {km}; "
            f"{row['censored_runs']} right-censored failures."
        )
    lines.extend([
        "",
        "These results describe only the configured maps, open-loop genomes, reward policy, "
        "hyperparameters, and budget. A smoke preset is for verifying the pipeline, not ranking "
        "algorithms. Fitness curves must not be compared across different reward policies. "
        "Environment-step totals supplement evaluation counts because successful episodes may stop early.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case_study(
    *,
    maps: tuple[str, ...] = ("spiral", "two_bridges"),
    fitnesses: tuple[str, ...] = ("euclidean", "geodesic", "checkpoints"),
    method: str = "ga",
    seeds: tuple[int, ...] = (0, 1, 2),
    population: int = 16,
    evaluations: int = 128,
) -> list[Run]:
    """Run matched reward studies; only physical outcomes should be compared."""
    runs = []
    for map_name in maps:
        for fitness_name in fitnesses:
            config = BenchmarkConfig(
                map_name=map_name, fitness_name=fitness_name, methods=(method,),
                seeds=seeds, population=population, evaluations=evaluations,
            )
            runs.extend(run_benchmark(config).runs)
    return runs


def case_study_figure(runs: list[Run], path: Path) -> None:
    """Compare fitness designs using reward-independent physical outcomes only."""
    rows = [run_row(run) for run in runs]
    maps = tuple(dict.fromkeys(row["map"] for row in rows))
    fitnesses = tuple(dict.fromkeys(row["fitness"] for row in rows))
    method = rows[0]["method"]
    evaluations = rows[0]["evaluations"]
    seeds_per_policy = len({row["seed"] for row in rows})
    colors = {"euclidean": "#d17a22", "geodesic": "#168579", "checkpoints": "#7655a6"}
    fig, axes = plt.subplots(2, len(maps), figsize=(16, 9), facecolor="white", squeeze=False)
    for column, map_name in enumerate(maps):
        map_rows = [row for row in rows if row["map"] == map_name]
        success = []
        distance_median = []
        distance_q1 = []
        distance_q3 = []
        for fitness_name in fitnesses:
            group = [row for row in map_rows if row["fitness"] == fitness_name]
            success.append(100 * sum(row["run_success"] for row in group) / len(group))
            median, q1, q3 = _median_iqr(row["final_distance"] for row in group)
            distance_median.append(median); distance_q1.append(q1); distance_q3.append(q3)
        x = np.arange(len(fitnesses))
        bars = axes[0, column].bar(x, success, color=[colors[name] for name in fitnesses])
        for bar, value in zip(bars, success):
            axes[0, column].text(
                bar.get_x() + bar.get_width() / 2, max(value, 1.5), f"{value:.0f}%",
                ha="center", va="bottom", fontsize=10,
            )
        axes[0, column].set_ylim(0, 100); axes[0, column].set_ylabel("run success rate (%)")
        axes[0, column].set_title(map_name.replace("_", " ").title())
        error = np.asarray([
            np.asarray(distance_median) - np.asarray(distance_q1),
            np.asarray(distance_q3) - np.asarray(distance_median),
        ])
        axes[1, column].bar(x, distance_median, yerr=error,
                            color=[colors[name] for name in fitnesses], capsize=4)
        axes[1, column].set_ylabel("final distance (median and IQR)")
        for ax in axes[:, column]:
            ax.set_xticks(x, fitnesses, rotation=15); ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        f"Fitness-design case study · {method.upper()} · {seeds_per_policy} seeds · "
        f"{evaluations} evaluations/run · shared physical outcomes only\n"
        "Geodesic grids and ordered checkpoints inject additional map-specific information",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, .94)); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white"); plt.close(fig)


def export_case_study(runs: list[Run], folder: Path) -> dict[str, Path]:
    raw_rows = [run_row(run) for run in runs]
    # Cross-policy exports deliberately omit reward values because scales differ.
    rows = [
        {key: value for key, value in row.items()
         if key not in {"best_fitness", "best_by_fitness_goal_reached"}}
        for row in raw_rows
    ]
    # Aggregation remains stratified by map, fitness, and method.
    summary = [
        {key: value for key, value in row.items() if not key.startswith("best_fitness_")}
        for row in aggregate_rows(raw_rows)
    ]
    paths = {
        "csv": folder / "fitness_case_study_runs.csv",
        "summary_csv": folder / "fitness_case_study_summary.csv",
        "json": folder / "fitness_case_study.json",
        "figure": folder / "fitness_case_study.png",
    }
    _write_csv(paths["csv"], rows); _write_csv(paths["summary_csv"], summary)
    paths["json"].write_text(json.dumps({"schema_version": 1, "runs": rows,
                                         "configuration": {
                                             "maps": sorted({row["map"] for row in rows}),
                                             "fitness_policies": sorted({row["fitness"] for row in rows}),
                                             "methods": sorted({row["method"] for row in rows}),
                                             "seeds": sorted({row["seed"] for row in rows}),
                                             "population": rows[0]["population"],
                                             "evaluations_per_run": rows[0]["evaluations"],
                                             "collision_penalty_parameter": 0.5,
                                             "smoothness_weight_parameter": 1.0,
                                         },
                                         "comparison_policy": (
                                             "Only reward-independent physical outcomes are exported; "
                                             "raw reward values are deliberately omitted."
                                         ),
                                         "aggregate": summary}, indent=2), encoding="utf-8")
    case_study_figure(runs, paths["figure"])
    return paths


def playback_for_run(config: BenchmarkConfig, run: Run) -> PlaybackData:
    world = World.from_map(get_map(config.map_name))
    fitness = make_fitness(config.fitness_name, collision_penalty=config.collision_penalty,
                           smoothness_weight=config.smoothness_weight)
    playback_config = PlaybackConfig(
        map_name=config.map_name, method=run.method, fitness_name=config.fitness_name,
        seed=run.seed, generations=config.batches, population=config.population,
        collision_penalty=config.collision_penalty, smoothness_weight=config.smoothness_weight,
    )
    return PlaybackData.from_run(playback_config, world, fitness, run)


def export_media(result: BenchmarkResult, folder: Path, *, episode_frames: int = 6) -> dict[str, Path]:
    """Export one real movement GIF per method plus matched GIF and PNG."""
    if not {"ga", "random", "local"}.issubset(result.config.methods):
        raise ValueError("Media export requires ga, random, and local")
    seed = result.config.seeds[0]
    selected_runs = {
        method: next(run for run in result.runs if run.method == method and run.seed == seed)
        for method in ("ga", "random", "local")
    }
    datasets = {
        method: playback_for_run(result.config, run) for method, run in selected_runs.items()
    }
    paths = {method: folder / f"{method}_movement.gif" for method in datasets}
    for method, data in datasets.items():
        export_method_gif(data, paths[method], episode_frames=episode_frames)
    paths["comparison_gif"] = folder / "matched_comparison.gif"
    paths["comparison_png"] = folder / "matched_comparison.png"
    export_comparison_gif(datasets, paths["comparison_gif"], episode_frames=episode_frames)
    export_comparison_png(datasets, paths["comparison_png"])
    return paths


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="Reproducible Bug Evolution Lab benchmark")
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--map", choices=tuple(MAPS))
    parser.add_argument("--fitness", choices=FITNESS_NAMES)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--population", type=int)
    parser.add_argument("--evaluations", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/task5"))
    parser.add_argument("--media", action="store_true", help="export four presentation GIFs and PNG")
    parser.add_argument("--case-study", action="store_true", help="run spiral/two-bridges reward study")
    parser.add_argument("--episode-frames", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    base = BenchmarkConfig.smoke() if args.preset == "smoke" else BenchmarkConfig.full()
    config = BenchmarkConfig(
        map_name=args.map or base.map_name,
        fitness_name=args.fitness or base.fitness_name,
        methods=tuple(args.methods) if args.methods else base.methods,
        seeds=tuple(args.seeds) if args.seeds else base.seeds,
        population=args.population or base.population,
        evaluations=args.evaluations or base.evaluations,
    )
    result = run_benchmark(config, record_snapshots=args.media)
    paths = export_benchmark(result, args.output)
    if args.media:
        paths.update(export_media(result, args.output, episode_frames=args.episode_frames))
    if args.case_study:
        study_runs = run_case_study(
            seeds=config.seeds, population=config.population, evaluations=config.evaluations
        )
        paths.update({f"case_{key}": value for key, value in export_case_study(
            study_runs, args.output
        ).items()})
    print(f"Completed {len(result.runs)} runs in {result.elapsed_seconds:.3f} seconds")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
