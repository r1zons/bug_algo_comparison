"""Cached, deterministic experiment playback for notebooks and static exports."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from .environment import Rollout, World, evaluate
from .experiment import Run, run_method
from .fitness import FITNESS_NAMES, FitnessPolicy, make_fitness
from .maps import MAPS, get_map
from .optimizers import METHODS, SearchConfig
from .visualization import field as draw_field


METHOD_DESCRIPTIONS = {
    "ga": "GA population: elite copies plus crossover-and-mutation offspring.",
    "random": "Random search: an independent fresh population in every batch.",
    "local": "Local search: candidate 0 is the incumbent after batch 0; the rest are neighbors.",
}


@dataclass(frozen=True)
class PlaybackConfig:
    """All settings that identify a cached teaching run."""

    map_name: str = "default"
    method: str = "ga"
    fitness_name: str = "geodesic"
    seed: int = 0
    generations: int = 8
    population: int = 16
    collision_penalty: float = 0.5
    smoothness_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.map_name not in MAPS:
            raise ValueError(f"Unknown map {self.map_name!r}")
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}")
        if self.fitness_name not in FITNESS_NAMES:
            raise ValueError(f"Unknown fitness {self.fitness_name!r}")
        if self.generations <= 0 or self.population <= 0:
            raise ValueError("generations and population must be positive")

    @property
    def label(self) -> str:
        return (
            f"{self.method} · {self.map_name} · {self.fitness_name} · "
            f"seed {self.seed} · {self.generations}×{self.population}"
        )


@dataclass(frozen=True)
class PlaybackSnapshot:
    """One bounded population snapshot; trajectories are replayed lazily."""

    generation: int
    evaluations: int
    candidate_genomes: np.ndarray
    candidate_scores: np.ndarray
    best_genome: np.ndarray
    best_fitness: float


@dataclass
class PlaybackData:
    """A recorded run plus lazy deterministic rollout caches."""

    config: PlaybackConfig
    world: World
    fitness: FitnessPolicy
    snapshots: tuple[PlaybackSnapshot, ...]
    history: tuple[dict[str, Any], ...]
    _rollout_cache: dict[tuple[int, int], Rollout] = field(default_factory=dict, repr=False)
    _best_cache: dict[int, Rollout] = field(default_factory=dict, repr=False)

    @classmethod
    def from_run(
        cls,
        config: PlaybackConfig,
        world: World,
        fitness: FitnessPolicy,
        run: Run,
    ) -> "PlaybackData":
        snapshots = []
        for snapshot in run.snapshots:
            if snapshot.candidate_genomes is None or snapshot.candidate_scores is None:
                raise ValueError("Playback requires snapshots with candidate genomes and scores")
            snapshots.append(PlaybackSnapshot(
                snapshot.generation,
                snapshot.evaluations,
                snapshot.candidate_genomes.copy(),
                snapshot.candidate_scores.copy(),
                snapshot.best_genome.copy(),
                float(snapshot.best_fitness),
            ))
        if not snapshots:
            raise ValueError("Playback requires at least one snapshot")
        return cls(config, world, fitness, tuple(snapshots), tuple(dict(row) for row in run.history))

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(snapshot.generation for snapshot in self.snapshots)

    @property
    def method_description(self) -> str:
        return METHOD_DESCRIPTIONS[self.config.method]

    @property
    def iteration_kind(self) -> str:
        return "generation" if self.config.method == "ga" else "batch"

    def iteration_label(self, snapshot_index: int) -> str:
        return f"{self.iteration_kind} {self.snapshot(snapshot_index).generation}"

    def snapshot(self, index: int) -> PlaybackSnapshot:
        if not 0 <= index < len(self.snapshots):
            raise IndexError(f"Snapshot index {index} outside 0..{len(self.snapshots)-1}")
        return self.snapshots[index]

    def snapshot_for_generation(self, generation: int) -> PlaybackSnapshot:
        for snapshot in self.snapshots:
            if snapshot.generation == generation:
                return snapshot
        raise KeyError(f"Generation {generation} was not recorded; available: {self.generations}")

    def snapshot_index(self, generation: int) -> int:
        try:
            return self.generations.index(generation)
        except ValueError as error:
            raise KeyError(f"Generation {generation} was not recorded") from error

    def history_at(self, snapshot_index: int) -> dict[str, Any]:
        generation = self.snapshot(snapshot_index).generation
        return self.history[generation]

    def candidate_rollout(self, snapshot_index: int, candidate_index: int) -> Rollout:
        snapshot = self.snapshot(snapshot_index)
        if not 0 <= candidate_index < len(snapshot.candidate_genomes):
            raise IndexError("candidate index outside the recorded population")
        key = (snapshot_index, candidate_index)
        if key not in self._rollout_cache:
            self._rollout_cache[key] = evaluate(
                self.world, snapshot.candidate_genomes[candidate_index], self.fitness
            )
        return self._rollout_cache[key]

    def best_rollout(self, snapshot_index: int) -> Rollout:
        if snapshot_index not in self._best_cache:
            self._best_cache[snapshot_index] = evaluate(
                self.world, self.snapshot(snapshot_index).best_genome, self.fitness
            )
        return self._best_cache[snapshot_index]

    def selected_candidate(self, snapshot_index: int) -> int:
        return int(np.argmax(self.snapshot(snapshot_index).candidate_scores))

    def best_so_far_source(
        self, snapshot_index: int
    ) -> tuple[int | None, int | None, int | None]:
        """Return snapshot, candidate, and exact ordinal where the stored best first appeared."""
        genome = self.snapshot(snapshot_index).best_genome
        for earlier_index in range(snapshot_index + 1):
            earlier = self.snapshot(earlier_index)
            matches = np.flatnonzero(np.all(earlier.candidate_genomes == genome, axis=1))
            if len(matches):
                candidate = int(matches[0])
                ordinal = earlier.generation * self.config.population + candidate + 1
                return earlier_index, candidate, ordinal
        return None, None, None

    def fastest_successful_source(
        self, snapshot_index: int
    ) -> tuple[int, int, int] | None:
        """Return the fastest successful evaluated candidate up to this snapshot."""
        fastest: tuple[int, int, int, int] | None = None
        for earlier_index in range(snapshot_index + 1):
            snapshot = self.snapshot(earlier_index)
            for candidate_index in range(len(snapshot.candidate_genomes)):
                rollout = self.candidate_rollout(earlier_index, candidate_index)
                if rollout.metrics is None or rollout.metrics.steps_to_goal is None:
                    continue
                ordinal = snapshot.generation * self.config.population + candidate_index + 1
                item = (rollout.metrics.steps_to_goal, ordinal, earlier_index, candidate_index)
                if fastest is None or item < fastest:
                    fastest = item
        if fastest is None:
            return None
        _, ordinal, earlier_index, candidate_index = fastest
        return earlier_index, candidate_index, ordinal

    def candidate_semantics(self, snapshot_index: int) -> dict[str, Any]:
        """Describe candidates without implying inheritance for non-genetic methods."""
        snapshot = self.snapshot(snapshot_index)
        best_index = self.selected_candidate(snapshot_index)
        common = {
            "iteration_kind": self.iteration_kind,
            "iteration": snapshot.generation,
            "best_by_fitness_candidate_index": best_index,
        }
        if self.config.method == "random":
            return {**common, "kind": "independent_samples", "independent": True}
        if self.config.method == "ga":
            return {**common, "kind": "population", "independent": False}
        if snapshot_index == 0:
            return {
                **common,
                "kind": "initial_candidates",
                "incumbent_before_candidate_index": None,
                "neighbor_candidate_indices": (),
                "proposed_candidate_index": best_index,
                "accepted": True,
                "incumbent_after_candidate_index": best_index,
            }
        incumbent_score = float(snapshot.candidate_scores[0])
        best_neighbor = int(np.argmax(snapshot.candidate_scores[1:])) + 1
        best_neighbor_score = float(snapshot.candidate_scores[best_neighbor])
        proposal_score = float(snapshot.candidate_scores[best_index])
        accepted = best_index != 0 and proposal_score > incumbent_score
        return {
            **common,
            "kind": "incumbent_and_neighbors",
            "incumbent_before_candidate_index": 0,
            "neighbor_candidate_indices": tuple(range(1, len(snapshot.candidate_genomes))),
            "best_neighbor_candidate_index": best_neighbor,
            "best_neighbor_score": best_neighbor_score,
            "proposed_candidate_index": best_index,
            "accepted": accepted,
            "incumbent_after_candidate_index": best_index if accepted else 0,
        }

    def ranked_candidates(self, snapshot_index: int, top_k: int | None = None) -> np.ndarray:
        order = np.argsort(self.snapshot(snapshot_index).candidate_scores)[::-1]
        if top_k is None or top_k <= 0:
            return order
        return order[:min(top_k, len(order))]

    def summary(self, snapshot_index: int, candidate_index: int | None = None) -> dict[str, Any]:
        snapshot = self.snapshot(snapshot_index)
        if candidate_index is None:
            candidate_index = self.selected_candidate(snapshot_index)
        rollout = self.candidate_rollout(snapshot_index, candidate_index)
        assert rollout.metrics is not None
        history = self.history_at(snapshot_index)
        best_index = self.selected_candidate(snapshot_index)
        best_rollout = self.best_rollout(snapshot_index)
        assert best_rollout.metrics is not None
        best_source_index,best_source_candidate,best_source_ordinal = self.best_so_far_source(snapshot_index)
        fastest = self.fastest_successful_source(snapshot_index)
        if fastest is None:
            fastest_fields = {
                "fastest_successful_snapshot_index": None,
                "fastest_successful_candidate_index": None,
                "fastest_successful_evaluation": None,
                "fastest_successful_steps_to_goal": None,
            }
        else:
            fastest_snapshot,fastest_candidate,fastest_ordinal = fastest
            fastest_rollout = self.candidate_rollout(fastest_snapshot,fastest_candidate)
            assert fastest_rollout.metrics is not None
            fastest_fields = {
                "fastest_successful_snapshot_index": fastest_snapshot,
                "fastest_successful_candidate_index": fastest_candidate,
                "fastest_successful_evaluation": fastest_ordinal,
                "fastest_successful_steps_to_goal": fastest_rollout.metrics.steps_to_goal,
            }
        return {
            "method": self.config.method,
            "map": self.config.map_name,
            "fitness": self.config.fitness_name,
            "seed": self.config.seed,
            "generation": snapshot.generation,
            "iteration": snapshot.generation,
            "iteration_kind": self.iteration_kind,
            "candidate": candidate_index,
            "selected_candidate_index": candidate_index,
            "score": float(snapshot.candidate_scores[candidate_index]),
            "selected_candidate_score": float(snapshot.candidate_scores[candidate_index]),
            "best_by_fitness_candidate_index": best_index,
            "best_by_fitness_score": float(snapshot.candidate_scores[best_index]),
            "best_so_far_score": snapshot.best_fitness,
            "best_so_far_snapshot_index": best_source_index,
            "best_so_far_candidate_index": best_source_candidate,
            "best_so_far_first_evaluation": best_source_ordinal,
            "best_so_far_goal_reached": best_rollout.metrics.goal_reached,
            "goal_reached": rollout.metrics.goal_reached,
            "selected_goal_reached": rollout.metrics.goal_reached,
            "steps_to_goal": rollout.metrics.steps_to_goal,
            "collisions": rollout.metrics.collision_count,
            "path_length": rollout.metrics.actual_path_length,
            "final_distance": rollout.metrics.final_distance,
            "evaluations": snapshot.evaluations,
            "environment_steps": history["environment_steps"],
            **fastest_fields,
        }

    def individual_summary(self, snapshot_index: int, candidate_index: int) -> dict[str, Any]:
        snapshot = self.snapshot(snapshot_index)
        genome = snapshot.candidate_genomes[candidate_index]
        summary = self.summary(snapshot_index, candidate_index)
        degrees = np.rad2deg(genome)
        preview = ", ".join(f"{value:+.1f}°" for value in degrees[:12])
        if len(degrees) > 12:
            preview += ", …"
        return {
            **summary,
            "genome_length": len(genome),
            "angles_preview": preview,
            "max_absolute_turn_degrees": float(np.max(np.abs(degrees))),
        }


class ExperimentCache:
    """In-memory cache; experiments run only through the explicit ``run`` method."""

    def __init__(self) -> None:
        self._runs: dict[PlaybackConfig, PlaybackData] = {}
        self.run_count = 0

    def register(self, data: PlaybackData) -> PlaybackData:
        self._runs[data.config] = data
        return data

    def get(self, config: PlaybackConfig) -> PlaybackData | None:
        return self._runs.get(config)

    def values(self) -> tuple[PlaybackData, ...]:
        return tuple(self._runs.values())

    def run(self, config: PlaybackConfig, *, force: bool = False) -> PlaybackData:
        if not force and config in self._runs:
            return self._runs[config]
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
        run = run_method(
            world,
            search,
            config.method,
            config.seed,
            config.generations,
            tuple(range(config.generations)),
            fitness,
        )
        self.run_count += 1
        return self.register(PlaybackData.from_run(config, world, fitness, run))

    def load(self, path: Path) -> PlaybackData:
        return self.register(load_playback_data(path))


def collision_points(rollout: Rollout) -> np.ndarray:
    """Return repeated positions caused by blocked movement steps."""
    if len(rollout.positions) < 2:
        return np.empty((0, 2))
    blocked = np.linalg.norm(np.diff(rollout.positions, axis=0), axis=1) < 1e-12
    return rollout.positions[1:][blocked]


def trajectory_figure(
    data: PlaybackData,
    snapshot_index: int,
    *,
    selected_candidate: int | None = None,
    show_all: bool = True,
    show_selected: bool = True,
    show_best_so_far: bool = True,
    show_collisions: bool = False,
    highlight_success: bool = True,
    show_checkpoints: bool = False,
    top_k: int | None = None,
):
    """Render one recorded generation without rerunning optimization."""
    snapshot = data.snapshot(snapshot_index)
    if selected_candidate is None:
        selected_candidate = data.selected_candidate(snapshot_index)
    fig, ax = plt.subplots(figsize=(9.4, 6.2), facecolor="white")
    draw_field(ax, data.world, show_checkpoints=show_checkpoints)
    if show_all:
        for candidate_index in data.ranked_candidates(snapshot_index, top_k):
            rollout = data.candidate_rollout(snapshot_index, int(candidate_index))
            color = "#4eaa78" if highlight_success and rollout.reached else "#8b96aa"
            ax.plot(rollout.positions[:, 0], rollout.positions[:, 1], color=color,
                    alpha=.23, lw=1.0)
    if show_selected:
        rollout = data.candidate_rollout(snapshot_index, selected_candidate)
        ax.plot(rollout.positions[:, 0], rollout.positions[:, 1], color="#2268a8",
                lw=2.7, label=f"Selected candidate {selected_candidate}")
        if show_collisions:
            points = collision_points(rollout)
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], marker="x", s=38,
                           color="#c43d3d", label="Blocked move", zorder=9)
    if show_best_so_far:
        best = data.best_rollout(snapshot_index)
        ax.plot(best.positions[:, 0], best.positions[:, 1], color="#138579", lw=2.3,
                ls="--", label="Best so far")
        if show_collisions and not show_selected:
            points = collision_points(best)
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], marker="x", s=38,
                           color="#c43d3d", label="Blocked move", zorder=9)
    ax.set_title(
        f"{data.config.method.upper()} · {data.config.map_name} · {data.config.fitness_name}\n"
        f"{data.iteration_kind} {snapshot.generation} · {snapshot.evaluations} evaluations",
        fontsize=14,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", framealpha=.94)
    fig.tight_layout()
    return fig


def progress_figure(data: PlaybackData, snapshot_index: int):
    """Plot fixed-fitness progress and shared outcomes against evaluations."""
    data.snapshot(snapshot_index)
    evaluations = np.asarray([row["evaluations"] for row in data.history])
    best_fitness = np.asarray([row["best_fitness"] for row in data.history])
    final_distance = np.asarray([row["final_distance"] for row in data.history])
    successes = np.asarray([row["success_rate"] * row["evaluations"] for row in data.history])
    selected_evaluations = data.snapshot(snapshot_index).evaluations
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2), facecolor="white")
    series = (
        (best_fitness, "Best fitness", "#2268a8"),
        (final_distance, "Representative final distance", "#c07828"),
        (successes, "Cumulative successful evaluations", "#26845b"),
    )
    for ax, (values, label, color) in zip(axes, series):
        ax.plot(evaluations, values, color=color, lw=2)
        ax.axvline(selected_evaluations, color="#4b4f56", ls="--", lw=1)
        ax.set_xlabel("evaluations");ax.set_ylabel(label);ax.grid(alpha=.2)
    fig.suptitle(
        f"Progress for one fixed fitness policy ({data.config.fitness_name}); "
        "raw fitness is not comparable across policies",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def summary_figure(data: PlaybackData, snapshot_index: int, candidate_index: int | None = None):
    """Create a presentation-ready run summary figure."""
    summary = data.summary(snapshot_index, candidate_index)
    fig = plt.figure(figsize=(12, 6.4), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.45))
    table_ax = fig.add_subplot(grid[0, :]);table_ax.axis("off")
    labels = (
        "method", "map", "fitness", "seed", "iteration_kind", "iteration", "evaluations",
        "selected_candidate_index", "selected_candidate_score",
        "best_by_fitness_candidate_index", "best_by_fitness_score",
        "best_so_far_score", "best_so_far_first_evaluation",
        "fastest_successful_evaluation", "fastest_successful_steps_to_goal",
        "selected_goal_reached", "steps_to_goal",
        "collisions", "path_length", "final_distance", "environment_steps",
    )
    cells = [[label.replace("_", " ").title(), f"{summary[label]:.3f}"
              if isinstance(summary[label], float) else str(summary[label])] for label in labels]
    table = table_ax.table(cellText=cells, colLabels=("Run field", "Value"),
                           cellLoc="left", colLoc="left", bbox=(0, 0, 1, .88))
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,.95)
    table_ax.set_title("Run and selected-individual summary", fontsize=15, pad=2)
    evaluations = np.asarray([row["evaluations"] for row in data.history])
    selected_evaluations = data.snapshot(snapshot_index).evaluations
    for ax, key, title, color, scale in (
        (fig.add_subplot(grid[1, 0]), "best_fitness", "Best fitness", "#2268a8", 1),
        (fig.add_subplot(grid[1, 1]), "final_distance", "Final distance", "#c07828", 1),
        (fig.add_subplot(grid[1, 2]), "success_rate", "Evaluated-genome success (%)", "#26845b", 100),
    ):
        ax.plot(evaluations, [scale*row[key] for row in data.history], color=color, lw=2)
        ax.axvline(selected_evaluations, color="#4b4f56", ls="--", lw=1)
        ax.set_xlabel("evaluations");ax.set_title(title);ax.grid(alpha=.2)
    return fig


def export_generation_png(data: PlaybackData, snapshot_index: int, path: Path, **kwargs: Any) -> None:
    fig = trajectory_figure(data, snapshot_index, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def export_summary_png(
    data: PlaybackData,
    snapshot_index: int,
    path: Path,
    candidate_index: int | None = None,
) -> None:
    fig = summary_figure(data, snapshot_index, candidate_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def save_playback_data(data: PlaybackData, path: Path) -> None:
    """Save compact genomes/scores/history; trajectories remain lazy after loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        "schema_version": 1,
        "config": asdict(data.config),
        "history": list(data.history),
    })
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        generations=np.asarray([snapshot.generation for snapshot in data.snapshots], dtype=int),
        evaluations=np.asarray([snapshot.evaluations for snapshot in data.snapshots], dtype=int),
        candidate_genomes=np.stack([snapshot.candidate_genomes for snapshot in data.snapshots]),
        candidate_scores=np.stack([snapshot.candidate_scores for snapshot in data.snapshots]),
        best_genomes=np.stack([snapshot.best_genome for snapshot in data.snapshots]),
        best_fitness=np.asarray([snapshot.best_fitness for snapshot in data.snapshots]),
    )


def load_playback_data(path: Path) -> PlaybackData:
    """Load a safe, non-pickle playback cache and reconstruct map/fitness metadata."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported playback cache schema")
        config = PlaybackConfig(**metadata["config"])
        world = World.from_map(get_map(config.map_name))
        fitness = make_fitness(
            config.fitness_name,
            collision_penalty=config.collision_penalty,
            smoothness_weight=config.smoothness_weight,
        )
        snapshots = tuple(
            PlaybackSnapshot(
                int(generation),
                int(evaluations),
                genomes.copy(),
                scores.copy(),
                best_genome.copy(),
                float(best_fitness),
            )
            for generation, evaluations, genomes, scores, best_genome, best_fitness in zip(
                archive["generations"], archive["evaluations"],
                archive["candidate_genomes"], archive["candidate_scores"],
                archive["best_genomes"], archive["best_fitness"],
            )
        )
    return PlaybackData(config, world, fitness, snapshots,
                        tuple(dict(row) for row in metadata["history"]))
