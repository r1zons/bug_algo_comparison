"""Declarative, deterministic lecture scenes rendered from recorded data."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
from typing import Any, Mapping

from matplotlib import animation, patches, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .animations import (
    METHOD_COLORS,
    METHOD_TITLES,
    trajectory_prefix,
    validate_matched_playback,
)
from .environment import simulate
from .operator_story import GeneticOperation, SyntheticOperation, load_operation_log, synthetic_operation
from .playback import PlaybackData, load_playback_data
from .visualization import PARENT_A_COLOR, PARENT_B_COLOR, MUTATION_COLOR, field


SCENE_KINDS = (
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
EXPORT_FORMATS = ("gif", "mp4")


@dataclass(frozen=True)
class LectureStyle:
    """Shared white-background 16:9 presentation style."""

    canvas_width: float = 12.8
    canvas_height: float = 7.2
    background: str = "white"
    text: str = "#20262d"
    trajectory: str = "#2268a8"
    population: str = "#6f8faa"
    highlight: str = "#168579"
    collision: str = "#c43d3d"
    secondary: str = "#7655a6"
    muted: str = "#8a96a3"
    font_family: str = "DejaVu Sans"

    def __post_init__(self) -> None:
        if not np.isclose(self.canvas_width / self.canvas_height, 16 / 9):
            raise ValueError("Lecture canvas must have a 16:9 aspect ratio")


@dataclass(frozen=True)
class SceneConfig:
    """Small declarative scene description; it contains no optimizer settings."""

    name: str
    kind: str
    title: str
    duration_seconds: float = 6.0
    fps: int = 6
    output_format: str = "gif"
    snapshot_index: int = 0
    candidate_index: int | None = None
    candidate_selector: str = "best_by_fitness"
    max_candidates: int = 12
    show_checkpoint_markers: bool = False
    data_path: str | None = None
    operation_log_path: str | None = None
    comparison_data_paths: Mapping[str, str] | None = None
    educational_message: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.title:
            raise ValueError("Scene name and title are required")
        if self.kind not in SCENE_KINDS:
            raise ValueError(f"Unknown scene kind {self.kind!r}")
        if self.duration_seconds <= 0 or self.fps <= 0:
            raise ValueError("Scene duration and fps must be positive")
        if self.output_format not in EXPORT_FORMATS:
            raise ValueError(f"output_format must be one of {EXPORT_FORMATS}")
        if self.candidate_selector not in ("best_by_fitness", "explicit"):
            raise ValueError("candidate_selector must be best_by_fitness or explicit")
        if self.candidate_selector == "explicit" and self.candidate_index is None:
            raise ValueError("explicit candidate selection requires candidate_index")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.kind == "method_comparison":
            if self.comparison_data_paths is None or set(self.comparison_data_paths) != {
                "ga", "random", "local"
            }:
                raise ValueError("method_comparison requires ga, random, and local data paths")
        if self.kind in ("evaluation_selection", "generation_transition") \
                and self.operation_log_path is None:
            raise ValueError(f"Scene {self.kind!r} requires operation_log_path")

    @property
    def frame_count(self) -> int:
        return max(2, int(round(self.duration_seconds * self.fps)))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SceneConfig":
        parsed = dict(values)
        if parsed.get("comparison_data_paths") is not None:
            parsed["comparison_data_paths"] = dict(parsed["comparison_data_paths"])
        return cls(**parsed)


def available_export_formats() -> tuple[str, ...]:
    formats = ["gif"]
    if animation.writers.is_available("ffmpeg"):
        formats.append("mp4")
    return tuple(formats)


def load_scene_configs(path: Path) -> tuple[SceneConfig, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes = tuple(SceneConfig.from_mapping(item) for item in payload["scenes"])
    if len({scene.name for scene in scenes}) != len(scenes):
        raise ValueError("Scene names must be unique")
    return scenes


def _apply_style(style: LectureStyle) -> None:
    plt.rcParams.update({
        "font.family": style.font_family,
        "text.color": style.text,
        "axes.labelcolor": style.text,
        "axes.titlecolor": style.text,
        "xtick.color": style.text,
        "ytick.color": style.text,
    })


def _save_movie(movie: animation.FuncAnimation, figure, path: Path, config: SceneConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.output_format == "gif":
        writer = animation.PillowWriter(fps=config.fps)
    else:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("MP4 export requested, but Matplotlib cannot find ffmpeg")
        writer = animation.FFMpegWriter(fps=config.fps, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
    movie.save(path, writer=writer, dpi=100)
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolated_trajectory_prefix(positions: np.ndarray, episode_step: float) -> np.ndarray:
    """Reveal an actual trajectory smoothly without changing its sampled path."""
    if len(positions) == 0:
        raise ValueError("A trajectory must contain at least one position")
    bounded = float(np.clip(episode_step, 0.0, len(positions) - 1))
    lower = int(np.floor(bounded))
    fraction = bounded - lower
    prefix = positions[:lower + 1].copy()
    if fraction > 0 and lower + 1 < len(positions):
        point = positions[lower] + fraction * (positions[lower + 1] - positions[lower])
        prefix = np.vstack((prefix, point))
    return prefix


def _snapshot_schedule(frame_count: int, snapshot_count: int) -> list[tuple[int, float]]:
    """Allocate motion plus a short final-state hold to every recorded batch."""
    if snapshot_count <= 0:
        raise ValueError("snapshot_count must be positive")
    base, extra = divmod(frame_count, snapshot_count)
    schedule: list[tuple[int, float]] = []
    for snapshot_index in range(snapshot_count):
        frames = max(1, base + int(snapshot_index < extra))
        for local_index in range(frames):
            phase = local_index / max(1, frames - 1)
            # The final 20% holds the evaluated state before the next batch.
            episode_fraction = min(1.0, phase / .8)
            schedule.append((snapshot_index, episode_fraction))
    return schedule[:frame_count]


def _selected_candidate(config: SceneConfig, data: PlaybackData) -> int:
    snapshot = data.snapshot(config.snapshot_index)
    if config.candidate_selector == "best_by_fitness":
        return data.selected_candidate(config.snapshot_index)
    assert config.candidate_index is not None
    if not 0 <= config.candidate_index < len(snapshot.candidate_genomes):
        raise IndexError("Configured candidate is outside the recorded population")
    return config.candidate_index


def _genome_digest(genome: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(genome, dtype=np.float64).tobytes()).hexdigest()


def _single_beetle_movie(
    config: SceneConfig, data: PlaybackData, style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    candidate = _selected_candidate(config, data)
    snapshot = data.snapshot(config.snapshot_index)
    genome = snapshot.candidate_genomes[candidate]
    rollout = data.candidate_rollout(config.snapshot_index, candidate)
    steps = np.linspace(0, data.world.steps, config.frame_count)
    figure, map_ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height),
                                  facecolor=style.background)
    figure.subplots_adjust(left=.09, right=.91, bottom=.11, top=.86)
    field(map_ax, data.world, show_checkpoints=config.show_checkpoint_markers)
    route, = map_ax.plot([], [], color=style.trajectory, lw=3, label="Executed trajectory")
    beetle, = map_ax.plot([], [], marker="o", ms=10, color=style.highlight, zorder=10)
    collision, = map_ax.plot([], [], marker="x", ms=10, mew=2, color=style.collision,
                             visible=False, label="Blocked move")
    degrees = np.rad2deg(genome)
    message = map_ax.text(.02, .97, "Genome = a sequence of relative turn commands",
                          transform=map_ax.transAxes, va="top", fontsize=13,
                          bbox={"boxstyle": "round,pad=.4", "fc": "white",
                                "ec": "#c7cdd3", "alpha": .96})
    status = map_ax.text(.985, .04, "", transform=map_ax.transAxes, va="bottom", ha="right",
                         fontsize=13, bbox={"boxstyle": "round,pad=.5", "fc": "white",
                                           "ec": "#c7cdd3", "alpha": .96})
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    figure.text(.5, .035, "Each gene changes heading; every unblocked step has the same length.",
                ha="center", fontsize=11)

    def draw(frame_index: int):
        requested_step_float = float(steps[frame_index])
        requested_step = int(np.floor(requested_step_float))
        position_index = min(requested_step, len(rollout.positions) - 1)
        prefix = interpolated_trajectory_prefix(rollout.positions, requested_step_float)
        route.set_data(prefix[:, 0], prefix[:, 1])
        beetle.set_data([prefix[-1, 0]], [prefix[-1, 1]])
        gene_index = 0 if position_index == 0 else min(position_index - 1, len(genome) - 1)
        blocked = position_index > 0 and np.allclose(
            rollout.positions[position_index], rollout.positions[position_index - 1]
        )
        collision.set_visible(blocked)
        if blocked:
            collision.set_data([prefix[-1, 0]], [prefix[-1, 1]])
        reached = rollout.reached and requested_step >= len(rollout.positions) - 1
        execution_text = (
            f"episode ended at step {len(rollout.positions) - 1}"
            if reached else f"executed steps {position_index}"
        )
        status.set_text(
            f"command {gene_index + 1:02d}/{len(genome)}   {degrees[gene_index]:+.1f}°\n"
            f"{execution_text}   ·   goal {'reached' if reached else 'not reached'}"
        )
        return route, beetle, collision, status, message

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    metadata = {
        "illustrative": False,
        "snapshot_index": config.snapshot_index,
        "candidate_index": candidate,
        "candidate_score": float(snapshot.candidate_scores[candidate]),
        "genome_sha256": _genome_digest(genome),
        "goal_reached": rollout.reached,
        "trajectory_points": len(rollout.positions),
    }
    return movie, figure, metadata


def _population_movie(
    config: SceneConfig, data: PlaybackData, style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    snapshot = data.snapshot(config.snapshot_index)
    count = min(config.max_candidates, len(snapshot.candidate_genomes))
    candidate_indices = tuple(range(count))
    rollouts = [data.candidate_rollout(config.snapshot_index, index) for index in candidate_indices]
    steps = np.linspace(0, data.world.steps, config.frame_count)
    figure, map_ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height),
                                  facecolor=style.background)
    field(map_ax, data.world, show_checkpoints=config.show_checkpoint_markers)
    lines = [map_ax.plot([], [], color=style.population, alpha=.3, lw=1.3)[0]
             for _ in candidate_indices]
    dots = [map_ax.plot([], [], marker="o", color=style.trajectory, alpha=.85, ms=5)[0]
            for _ in candidate_indices]
    status = map_ax.text(.985, .04, "", transform=map_ax.transAxes, ha="right", va="bottom",
                         fontsize=13, bbox={"boxstyle": "round,pad=.5", "fc": "white",
                                           "ec": "#c7cdd3", "alpha": .96})
    map_ax.text(.02, .97, "One genome per beetle · identical physics",
                transform=map_ax.transAxes, va="top", fontsize=13,
                bbox={"boxstyle": "round,pad=.4", "fc": "white", "ec": "#c7cdd3"})
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    figure.subplots_adjust(left=.09, right=.91, bottom=.11, top=.86)
    figure.text(.5, .035, "Actual independently sampled genomes · no crossover yet",
                ha="center", fontsize=11)

    def draw(frame_index: int):
        requested_step_float = float(steps[frame_index])
        requested_step = int(np.floor(requested_step_float))
        active = 0; successes = 0
        for rollout, line, dot in zip(rollouts, lines, dots):
            position_index = min(requested_step, len(rollout.positions) - 1)
            prefix = interpolated_trajectory_prefix(rollout.positions, requested_step_float)
            line.set_data(prefix[:, 0], prefix[:, 1])
            dot.set_data([prefix[-1, 0]], [prefix[-1, 1]])
            active += int(position_index < len(rollout.positions) - 1)
            successes += int(rollout.reached and position_index == len(rollout.positions) - 1)
        status.set_text(
            f"step {requested_step:02d}/{data.world.steps}\n"
            f"{count} different genomes · {successes} reached goal"
        )
        return (*lines, *dots, status)

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    metadata = {
        "illustrative": False,
        "snapshot_index": config.snapshot_index,
        "candidate_indices": candidate_indices,
        "genome_sha256": [
            _genome_digest(snapshot.candidate_genomes[index]) for index in candidate_indices
        ],
        "successful_candidates": sum(rollout.reached for rollout in rollouts),
    }
    return movie, figure, metadata


def _draw_gene_row(ax, genome: np.ndarray, y: float, label: str, colors: list[str],
                   visible: int | None = None) -> None:
    values = np.rad2deg(genome)
    limit = len(values) if visible is None else min(visible, len(values))
    for index in range(limit):
        ax.add_patch(patches.Rectangle((index - .43, y - .28), .86, .56,
                                       facecolor=colors[index], edgecolor="white", lw=1.4))
        ax.text(index, y, f"{values[index]:+.0f}°", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
    ax.text(-.65, y, label, ha="right", va="center", fontsize=12, fontweight="bold")


def _synthetic_movie(
    config: SceneConfig, operation: SyntheticOperation, style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    figure, ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height),
                              facecolor=style.background)
    progress = np.linspace(0, 1, config.frame_count)
    inherited = [PARENT_A_COLOR if index < operation.cut else PARENT_B_COLOR
                 for index in range(len(operation.child))]
    mutated = inherited.copy()
    for index in operation.mutated_indices:
        mutated[index] = MUTATION_COLOR

    def draw(frame_index: int):
        ax.clear(); phase = float(progress[frame_index])
        ax.set_xlim(-2.25, len(operation.child) - .25); ax.set_ylim(-1.35, 4.15)
        ax.axis("off")
        _draw_gene_row(ax, operation.parent_a, 3.0, "Parent A",
                       [PARENT_A_COLOR] * len(operation.parent_a))
        _draw_gene_row(ax, operation.parent_b, 2.0, "Parent B",
                       [PARENT_B_COLOR] * len(operation.parent_b))
        message = "Two illustrative parent instruction sequences"
        if phase >= .2:
            ax.axvline(operation.cut - .5, ymin=.34, ymax=.86, color=style.text, ls="--", lw=2)
            ax.text(operation.cut - .5, 3.65, f"one-point cut at index {operation.cut}",
                    ha="center", fontsize=11)
            message = "Choose one crossover cut"
        if phase >= .4:
            reveal = max(1, int(np.ceil((phase - .4) / .25 * len(operation.child))))
            _draw_gene_row(ax, operation.child_before_mutation, 1.0,
                           "Child after crossover", inherited, visible=reveal)
            message = "Copy genes from A before the cut, then from B"
        if phase >= .7:
            _draw_gene_row(ax, operation.child, 0.0, "Final child", mutated)
            for index in operation.mutated_indices:
                ax.annotate("mutation", xy=(index, .28), xytext=(index, .62), ha="center",
                            color=MUTATION_COLOR, fontsize=9,
                            arrowprops={"arrowstyle": "->", "color": MUTATION_COLOR})
            message = "Apply the explicitly marked mutations"
        ax.set_title(config.title + "\nIllustrative deterministic example", fontsize=18,
                     fontweight="bold", pad=12)
        ax.text((len(operation.child) - 1) / 2, -0.72, message, ha="center", fontsize=13)
        ax.text((len(operation.child) - 1) / 2, -1.08,
                "Genes are steering commands—not geometrically connected trajectory segments.",
                ha="center", fontsize=11)
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    metadata = {
        "illustrative": True,
        "cut": operation.cut,
        "mutated_indices": operation.mutated_indices,
        "parent_a_radians": operation.parent_a.tolist(),
        "parent_b_radians": operation.parent_b.tolist(),
        "child_radians": operation.child.tolist(),
    }
    return movie, figure, metadata


def _operation_for_generation(
    operations: list[GeneticOperation], generation: int, *, require_mutation: bool = False
) -> GeneticOperation:
    candidates = [event for event in operations
                  if event.generation == generation and event.kind == "offspring"]
    if require_mutation:
        candidates = [event for event in candidates if event.mutated_indices]
    if not candidates:
        raise ValueError(f"No suitable offspring operation recorded for generation {generation}")
    return candidates[0]


def _selected_parent_ids(operations: list[GeneticOperation], generation: int, limit: int = 3) -> tuple[int, ...]:
    counts: Counter[int] = Counter()
    for event in operations:
        if event.generation != generation or event.kind != "offspring":
            continue
        counts[event.parent_a_id] += 1
        if event.parent_b_id is not None:
            counts[event.parent_b_id] += 1
    return tuple(parent for parent, _ in counts.most_common(limit))


def _selection_movie(
    config: SceneConfig, data: PlaybackData, operations: list[GeneticOperation], style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    snapshot = data.snapshot(config.snapshot_index)
    count = min(config.max_candidates, len(snapshot.candidate_genomes))
    selected = _selected_parent_ids(operations, snapshot.generation + 1)
    rollouts = [data.candidate_rollout(config.snapshot_index, index) for index in range(count)]
    ranked = list(data.ranked_candidates(config.snapshot_index))
    rank = {int(candidate): position + 1 for position, candidate in enumerate(ranked)}
    progress = np.linspace(0, 1, config.frame_count)
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(2.25, .75))
    map_ax = figure.add_subplot(grid[0, 0]); score_ax = figure.add_subplot(grid[0, 1])
    figure.subplots_adjust(left=.055, right=.97, bottom=.12, top=.86, wspace=.18)
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    message_artist = figure.text(.5, .04, "", ha="center", fontsize=13, fontweight="bold")

    def draw(frame_index: int):
        phase = float(progress[frame_index]); map_ax.clear(); score_ax.clear()
        field(map_ax, data.world, show_checkpoints=False)
        path_fraction = min(1.0, phase / .38)
        for candidate, rollout in enumerate(rollouts):
            path = interpolated_trajectory_prefix(
                rollout.positions, path_fraction * (len(rollout.positions) - 1)
            )
            chosen = candidate in selected and phase >= .62
            map_ax.plot(path[:, 0], path[:, 1], color=style.highlight if chosen else style.population,
                        alpha=.95 if chosen else .26, lw=3 if chosen else 1.2)
            if chosen:
                map_ax.plot(path[-1, 0], path[-1, 1], "o", color=style.highlight, ms=8)
                map_ax.annotate(f"parent {selected.index(candidate)+1}", path[-1],
                                xytext=(6, 6), textcoords="offset points", fontsize=10,
                                color=style.highlight, fontweight="bold")
        score_ax.set_title("Closest-approach score", fontsize=14, fontweight="bold")
        score_ax.set_xlabel("− min distance over trajectory")
        score_ax.set_yticks(range(count), [f"beetle {index}" for index in range(count)])
        score_ax.grid(axis="x", alpha=.18)
        if phase >= .38:
            scores = snapshot.candidate_scores[:count]
            colors = [style.highlight if index in selected and phase >= .62 else style.population
                      for index in range(count)]
            sizes = [95 if index in selected and phase >= .62 else 42 for index in range(count)]
            score_ax.scatter(scores, range(count), c=colors, s=sizes, zorder=3)
            for candidate in selected:
                if candidate < count and phase >= .62:
                    score_ax.annotate(f"rank {rank[candidate]}",
                                      (snapshot.candidate_scores[candidate], candidate),
                                      xytext=(7, 0), textcoords="offset points", va="center",
                                      fontsize=10, color=style.highlight)
        if phase < .38:
            message = "1  Simulate every candidate"
        elif phase < .62:
            message = "2  Score the closest straight-line approach anywhere in the trajectory"
        else:
            message = "3  Better candidates are more likely to be selected as parents"
        message_artist.set_text(message)
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    return movie, figure, {
        "illustrative": False,
        "snapshot_index": config.snapshot_index,
        "selected_parent_ids": selected,
        "selection_source": "actual traced tournament selections for the next generation",
        "fitness_explanation": (
            "fitness = negative minimum Euclidean goal distance over the complete trajectory; "
            "this is not terminal-position distance"
        ),
    }


def _transition_movie(
    config: SceneConfig, data: PlaybackData, operations: list[GeneticOperation], style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    if len(data.snapshots) < 2:
        raise ValueError("generation_transition requires at least two snapshots")
    first_index = config.snapshot_index
    second_index = first_index + 1
    first = data.snapshot(first_index); second = data.snapshot(second_index)
    operation = _operation_for_generation(operations, second.generation, require_mutation=True)
    parent_ids = (operation.parent_a_id, int(operation.parent_b_id))
    first_paths = [data.candidate_rollout(first_index, index).positions
                   for index in range(min(config.max_candidates, len(first.candidate_genomes)))]
    second_paths = [data.candidate_rollout(second_index, index).positions
                    for index in range(min(config.max_candidates, len(second.candidate_genomes)))]
    progress = np.linspace(0, 1, config.frame_count)
    figure, ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    figure.subplots_adjust(left=.09, right=.91, bottom=.12, top=.86)
    figure.suptitle(config.title, fontsize=18, fontweight="bold")

    def draw(frame_index: int):
        phase = float(progress[frame_index]); ax.clear(); field(ax, data.world)
        if phase < .58:
            local = min(1.0, phase / .28)
            for candidate, path in enumerate(first_paths):
                prefix = interpolated_trajectory_prefix(path, local * (len(path) - 1))
                chosen = candidate in parent_ids and phase >= .28
                ax.plot(prefix[:, 0], prefix[:, 1],
                        color=style.highlight if chosen else style.population,
                        alpha=.95 if chosen else .24, lw=3 if chosen else 1.1)
                if chosen:
                    ax.annotate("selected parent", path[-1], xytext=(6, 6),
                                textcoords="offset points", color=style.highlight,
                                fontsize=10, fontweight="bold")
            label = f"Generation {first.generation}: evaluate the population"
            if phase >= .28:
                label = "Selection: two recorded parents are chosen"
            if phase >= .44:
                label = (f"Create offspring: one-point cut at gene {operation.cut}, "
                         f"then {len(operation.mutated_indices)} mutated genes")
                ax.text(.5, .16, "selection  →  crossover  →  mutation",
                        transform=ax.transAxes, ha="center", fontsize=16, fontweight="bold",
                        bbox={"boxstyle": "round,pad=.6", "fc": "white", "ec": style.highlight,
                              "alpha": .96})
        else:
            local = min(1.0, (phase - .58) / .32)
            for path in second_paths:
                prefix = interpolated_trajectory_prefix(path, local * (len(path) - 1))
                ax.plot(prefix[:, 0], prefix[:, 1], color=style.trajectory,
                        alpha=.3, lw=1.2)
            child_path = data.candidate_rollout(second_index, operation.candidate_index).positions
            child_prefix = interpolated_trajectory_prefix(
                child_path, local * (len(child_path) - 1)
            )
            ax.plot(child_prefix[:, 0], child_prefix[:, 1],
                    color=MUTATION_COLOR, lw=3, label="recorded offspring")
            ax.legend(loc="lower right", fontsize=10)
            label = f"Generation {second.generation}: evaluate the complete new population"
            ax.text(.5, .10,
                    "Highlighted: one recorded offspring example; every other offspring has its own recorded operations.",
                    transform=ax.transAxes, ha="center", fontsize=9.5,
                    bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#d3d8dd",
                          "alpha": .94})
        ax.text(.5, .97, label, transform=ax.transAxes, ha="center", va="top", fontsize=14,
                fontweight="bold", bbox={"boxstyle": "round,pad=.4", "fc": "white",
                                         "ec": "#c7cdd3", "alpha": .96})
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    return movie, figure, {
        "illustrative": False,
        "from_generation": first.generation,
        "to_generation": second.generation,
        "operation_id": operation.operation_id,
        "parent_ids": parent_ids,
        "child_candidate_index": operation.candidate_index,
        "crossover_cut": operation.cut,
        "mutated_indices": operation.mutated_indices,
    }


def _multi_generation_movie(
    config: SceneConfig, data: PlaybackData, style: LectureStyle, *, show_families: bool = False
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    count = min(config.max_candidates, data.config.population)
    rollouts = {
        (snapshot_index, candidate): data.candidate_rollout(snapshot_index, candidate)
        for snapshot_index in range(len(data.snapshots)) for candidate in range(count)
    }
    paths = {key: rollout.positions for key, rollout in rollouts.items()}
    schedule = _snapshot_schedule(config.frame_count, len(data.snapshots))
    family_counts: list[dict[str, int]] = []
    for snapshot_index in range(len(data.snapshots)):
        counts = Counter(
            _route_family(paths[(snapshot_index, candidate)], data.world)
            for candidate in range(count)
        )
        counts["physical goal successes"] = sum(
            int(rollouts[(snapshot_index, candidate)].reached) for candidate in range(count)
        )
        family_counts.append(dict(counts))
    figure, ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    figure.subplots_adjust(left=.08, right=.92, bottom=.13, top=.85)
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    if show_families:
        figure.text(.5, .045, "Opening color ≠ physical goal success.", ha="center",
                    fontsize=10.5, fontweight="bold", color=style.text)

    def draw(frame_index: int):
        snapshot_index, fraction = schedule[frame_index]
        snapshot = data.snapshot(snapshot_index); ax.clear(); field(ax, data.world)
        for candidate in range(count):
            path = paths[(snapshot_index, candidate)]
            prefix = interpolated_trajectory_prefix(path, fraction * (len(path) - 1))
            family = _route_family(path, data.world)
            if show_families:
                color = {"lower crossing": style.trajectory, "upper crossing": style.secondary}.get(
                    family, style.muted
                )
                alpha = .48 if family in ("lower crossing", "upper crossing") else .14
                width = 1.6 if family in ("lower crossing", "upper crossing") else 1.0
            else:
                color = style.population
                alpha = .25; width = 1.2
            ax.plot(prefix[:, 0], prefix[:, 1], color=color, alpha=alpha, lw=width)
        current_index = data.selected_candidate(snapshot_index)
        current_rollout = data.candidate_rollout(snapshot_index, current_index)
        current = interpolated_trajectory_prefix(
            current_rollout.positions, fraction * (len(current_rollout.positions) - 1)
        )
        ax.plot(current[:, 0], current[:, 1], color=style.text, lw=2.2,
                label="current generation leader")
        best_rollout = data.best_rollout(snapshot_index)
        best = best_rollout.positions
        ax.plot(best[:, 0], best[:, 1], color=style.highlight, lw=3, ls="--",
                label="best-so-far by fitness")
        if show_families:
            ax.legend(handles=[
                Line2D([0], [0], color=style.trajectory, lw=3, label="lower barrier crossing"),
                Line2D([0], [0], color=style.secondary, lw=3, label="upper barrier crossing"),
                Line2D([0], [0], color=style.text, lw=2.2,
                       label="current generation leader"),
                Line2D([0], [0], color=style.highlight, lw=3, ls="--",
                       label="best-so-far by fitness"),
            ], loc="upper left", fontsize=10)
        else:
            ax.legend(loc="upper left", fontsize=10)
        status = (f"generation {snapshot.generation} · {snapshot.evaluations} evaluations\n"
                  f"current leader {snapshot.candidate_scores[current_index]:.2f} · "
                  f"best-so-far {snapshot.best_fitness:.2f}")
        ax.text(.985, .035, status, transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                bbox={"boxstyle": "round,pad=.4", "fc": "white", "ec": "#c7cdd3"})
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    return movie, figure, {
        "illustrative": False,
        "generations_shown": list(data.generations),
        "best_fitness_by_generation": [snapshot.best_fitness for snapshot in data.snapshots],
        "route_family_counts": family_counts if show_families else None,
    }


def _barrier_crossing_y(positions: np.ndarray, barrier_x: float = 50.0) -> float | None:
    """Interpolate the first left-to-right crossing of the barrier centre line."""
    for first, second in zip(positions[:-1], positions[1:]):
        if first[0] < barrier_x <= second[0] and second[0] > first[0]:
            fraction = (barrier_x - first[0]) / (second[0] - first[0])
            return float(first[1] + fraction * (second[1] - first[1]))
    return None


def _route_family(positions: np.ndarray, world=None) -> str:
    """Classify a physical centre-line crossing, independently of goal success."""
    y = _barrier_crossing_y(positions)
    if y is None:
        return "did not cross"
    radius = 1.1 if world is None else float(world.radius)
    if 18.0 + radius <= y <= 28.0 - radius:
        return "lower crossing"
    if 52.0 + radius <= y <= 62.0 - radius:
        return "upper crossing"
    return "other crossing"


def _comparison_movie(
    config: SceneConfig, datasets: Mapping[str, PlaybackData], style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    validate_matched_playback(datasets)
    ordered = [datasets[method] for method in ("ga", "random", "local")]
    count = min(config.max_candidates, ordered[0].config.population)
    paths = {
        (data.config.method, snapshot_index, candidate):
            data.candidate_rollout(snapshot_index, candidate).positions
        for data in ordered for snapshot_index in range(len(data.snapshots))
        for candidate in range(data.config.population)
    }
    schedule = _snapshot_schedule(config.frame_count, len(ordered[0].snapshots))
    figure, axes = plt.subplots(1, 3, figsize=(style.canvas_width, style.canvas_height),
                                facecolor="white", sharex=True, sharey=True)
    figure.subplots_adjust(left=.045, right=.985, bottom=.14, top=.82, wspace=.12)
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    figure.text(.5, .055,
                "Dashed black = best-so-far · GA population · random fresh batch · local incumbent + neighbors",
                ha="center", fontsize=11.5, color=style.text)

    def draw(frame_index: int):
        snapshot_index, fraction = schedule[frame_index]
        for ax, data in zip(axes, ordered):
            ax.clear(); field(ax, data.world)
            ax.set_xlabel("")
            if ax is not axes[0]:
                ax.set_ylabel("")
            snapshot = data.snapshot(snapshot_index)
            semantics = data.candidate_semantics(snapshot_index)
            if data.config.method == "random":
                candidates = tuple(range(count))
            else:
                ranked = [int(index) for index in data.ranked_candidates(snapshot_index)]
                if data.config.method == "local":
                    ranked = [0] + [index for index in ranked if index != 0]
                candidates = tuple(ranked[:count])
            for candidate in candidates:
                path = paths[(data.config.method, snapshot_index, candidate)]
                prefix = interpolated_trajectory_prefix(path, fraction * (len(path) - 1))
                color = METHOD_COLORS[data.config.method]; alpha = .22; width = 1.0
                if data.config.method == "local" and candidate == 0:
                    color = style.trajectory; alpha = .85; width = 2.2
                if (data.config.method == "local" and snapshot_index > 0
                        and candidate == semantics["proposed_candidate_index"]):
                    color = style.highlight if semantics["accepted"] else style.collision
                    alpha = .9; width = 2.2
                ax.plot(prefix[:, 0], prefix[:, 1], color=color, alpha=alpha, lw=width)
            best_path = data.best_rollout(snapshot_index).positions
            best = interpolated_trajectory_prefix(
                best_path, fraction * (len(best_path) - 1)
            )
            ax.plot(best[:, 0], best[:, 1], color=style.text, lw=2.5, ls="--")
            if data.config.method == "ga":
                detail = "evolving population"
            elif data.config.method == "random":
                detail = "independent new batch"
            elif snapshot_index == 0:
                detail = "establish incumbent"
            else:
                detail = "best neighbor accepted" if semantics["accepted"] else "best neighbor rejected"
            ax.set_title(f"{METHOD_TITLES[data.config.method]}\n{detail}", fontsize=13,
                         fontweight="bold")
            ax.text(.5, .025, f"{data.iteration_kind} {snapshot.generation} · "
                    f"{snapshot.evaluations} evaluations", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=10,
                    bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": "#d3d8dd",
                          "alpha": .92})
            if data.config.method == "local" and snapshot_index > 0:
                ax.text(.5, .955, "blue: incumbent · green/red: proposed neighbor",
                        transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                        bbox={"boxstyle": "round,pad=.2", "fc": "white", "ec": "#d3d8dd",
                              "alpha": .9})
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    return movie, figure, {
        "illustrative": False,
        "methods": [data.config.method for data in ordered],
        "matched_configuration": asdict(ordered[0].config) | {"method": "matched"},
        "candidate_semantics": {
            "ga": "population with inheritance",
            "random": "independent batches",
            "local": "incumbent plus mutated neighbors",
        },
    }


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def _corridor_reference_trace(data: PlaybackData):
    """Physics-valid explanatory route through map checkpoints; never used as optimizer input."""
    world = data.world
    x, y = world.start
    heading = atan2(world.goal[1] - y, world.goal[0] - x)
    targets = [*world.checkpoints, world.goal]
    target_index = 0; turns = []
    for _ in range(world.steps):
        while target_index < len(targets) - 1 and hypot(
            x - targets[target_index][0], y - targets[target_index][1]
        ) <= 2.0:
            target_index += 1
        target = targets[target_index]
        desired = atan2(target[1] - y, target[0] - x)
        turn = float(np.clip(_wrap_angle(desired - heading), -world.max_turn, world.max_turn))
        turns.append(turn); heading += turn
        nx = x + world.step_length * cos(heading); ny = y + world.step_length * sin(heading)
        if not world.blocked_scalar(x, y, nx, ny):
            x, y = nx, ny
    return simulate(world, np.asarray(turns))


def _corridor_reference_positions(data: PlaybackData) -> np.ndarray:
    """Compatibility wrapper returning the reference trace positions."""
    return _corridor_reference_trace(data).positions


def _spiral_movie(
    config: SceneConfig, data: PlaybackData, style: LectureStyle
) -> tuple[animation.FuncAnimation, Any, dict[str, Any]]:
    count = min(config.max_candidates, data.config.population)
    final_index = len(data.snapshots) - 1
    paths = [data.candidate_rollout(final_index, candidate).positions for candidate in range(count)]
    best = data.best_rollout(final_index)
    reference_trace = _corridor_reference_trace(data)
    reference = reference_trace.positions
    distances = np.linalg.norm(reference - np.asarray(data.world.goal), axis=1)
    increases = np.flatnonzero(np.diff(distances) > 1e-6)
    progress = np.linspace(0, 1, config.frame_count)
    figure, ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    figure.subplots_adjust(left=.12, right=.88, bottom=.11, top=.85)
    figure.suptitle(config.title, fontsize=18, fontweight="bold")

    def draw(frame_index: int):
        phase = float(progress[frame_index]); ax.clear(); field(ax, data.world)
        for radius in (10, 20, 30, 40):
            ax.add_patch(patches.Circle(data.world.goal, radius, fill=False, ls=":", lw=1,
                                        ec="#d69a55", alpha=.35))
        episode_fraction = min(1.0, phase / .55)
        for path in paths:
            prefix = interpolated_trajectory_prefix(
                path, episode_fraction * (len(path) - 1)
            )
            ax.plot(prefix[:, 0], prefix[:, 1], color="#d17a22", alpha=.18, lw=1.1)
        best_prefix = interpolated_trajectory_prefix(
            best.positions, episode_fraction * (len(best.positions) - 1)
        )
        ax.plot(best_prefix[:, 0], best_prefix[:, 1], color="#d17a22", lw=3,
                label="best under closest-approach fitness")
        ax.plot([data.world.start[0], data.world.goal[0]],
                [data.world.start[1], data.world.goal[1]], color="#d17a22", ls="--", lw=1.5,
                alpha=.75, label="straight-line direction")
        message = r"Fitness = $-\min_t$ straight-line distance: one close moment can dominate."
        if phase >= .55:
            ax.plot(reference[:, 0], reference[:, 1], color=style.muted, lw=2, ls="--",
                    alpha=.8, label=("physics-valid corridor reference" +
                                     (" (reaches goal)" if reference_trace.reached
                                      else " (incomplete)")))
            if len(increases):
                for index in increases:
                    ax.plot(reference[index:index + 2, 0], reference[index:index + 2, 1],
                            color=MUTATION_COLOR, lw=4)
                first = int(increases[0])
                ax.annotate("detour segment begins:\ndistance initially increases",
                            reference[first], xytext=(24, -55), textcoords="offset points",
                            fontsize=10.5, color=MUTATION_COLOR,
                            arrowprops={"arrowstyle": "->", "color": MUTATION_COLOR},
                            bbox={"boxstyle": "round,pad=.3", "fc": "white",
                                  "ec": MUTATION_COLOR, "alpha": .95})
            message = "A physically valid corridor detour can first move farther from the goal."
        if phase >= .78:
            nearest = int(np.argmin(np.linalg.norm(best.positions - np.asarray(data.world.goal), axis=1)))
            ax.annotate("looks close\nbut is separated by a wall", best.positions[nearest],
                        xytext=(-105, 42), textcoords="offset points", fontsize=11,
                        arrowprops={"arrowstyle": "->", "color": style.text},
                        bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "#c7cdd3"})
        ax.text(.5, .97, message, transform=ax.transAxes, ha="center", va="top", fontsize=13,
                fontweight="bold", bbox={"boxstyle": "round,pad=.4", "fc": "white",
                                         "ec": "#c7cdd3", "alpha": .96})
        ax.text(.5, .012,
                "One deterministic failed GA run illustrates the pressure; it does not prove GA cannot solve this map.",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
                bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": "#d3d8dd",
                      "alpha": .94})
        ax.legend(loc="upper left", bbox_to_anchor=(.01, .86), fontsize=9, framealpha=.95)
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000 / config.fps, blit=False)
    return movie, figure, {
        "illustrative": False,
        "snapshot_index": final_index,
        "actual_euclidean_candidates": count,
        "run_goal_reached": any(row["goal_reached"] for row in data.history),
        "best_minimum_euclidean_distance": float(np.min(
            np.linalg.norm(best.positions - np.asarray(data.world.goal), axis=1)
        )),
        "fitness_definition": "negative minimum Euclidean distance over the full trajectory",
        "fitness_is_final_position_distance": False,
        "corridor_reference": "deterministic checkpoint-following trajectory for explanation only",
        "corridor_has_distance_increase": bool(len(increases)),
        "corridor_reference_reached_goal": reference_trace.reached,
        "corridor_reference_collisions": reference_trace.collisions,
        "interpretation_scope": (
            "illustrates selection pressure in one deterministic run; does not establish "
            "general optimizer impossibility"
        ),
    }


def export_scene(
    config: SceneConfig,
    output_dir: Path,
    *,
    data: PlaybackData | None = None,
    datasets: Mapping[str, PlaybackData] | None = None,
    operations: list[GeneticOperation] | None = None,
    source_paths: Mapping[str, str] | None = None,
    style: LectureStyle = LectureStyle(),
) -> tuple[Path, Path]:
    """Render one scene. This function never creates or runs an optimizer."""
    _apply_style(style)
    if config.output_format not in available_export_formats():
        raise RuntimeError(f"Export format {config.output_format!r} is unavailable")
    if config.kind == "synthetic_crossover":
        movie, figure, source = _synthetic_movie(config, synthetic_operation(), style)
        data_metadata = None
    elif config.kind == "method_comparison":
        if datasets is None:
            raise ValueError("method_comparison requires recorded datasets")
        movie, figure, source = _comparison_movie(config, datasets, style)
        data_metadata = {method: asdict(item.config) for method, item in datasets.items()}
    else:
        if data is None:
            raise ValueError(f"Scene {config.kind!r} requires recorded PlaybackData")
        if config.kind == "single_beetle":
            movie, figure, source = _single_beetle_movie(config, data, style)
        elif config.kind == "initial_population":
            movie, figure, source = _population_movie(config, data, style)
        elif config.kind == "evaluation_selection":
            movie, figure, source = _selection_movie(config, data, operations or [], style)
        elif config.kind == "generation_transition":
            movie, figure, source = _transition_movie(config, data, operations or [], style)
        elif config.kind == "multi_generation":
            movie, figure, source = _multi_generation_movie(config, data, style)
        elif config.kind == "route_families":
            movie, figure, source = _multi_generation_movie(
                config, data, style, show_families=True
            )
        elif config.kind == "spiral_limitation":
            movie, figure, source = _spiral_movie(config, data, style)
        else:  # guarded by SceneConfig validation
            raise ValueError(f"Unsupported scene kind {config.kind!r}")
        data_metadata = asdict(data.config)
    output_path = output_dir / f"{config.name}.{config.output_format}"
    metadata_path = output_dir / f"{config.name}.json"
    _save_movie(movie, figure, output_path, config)
    metadata = {
        "schema_version": 1,
        "scene": asdict(config),
        "style": asdict(style),
        "source_playback": data_metadata,
        "source_data_paths": dict(source_paths or {}),
        "source_details": source,
        "export": {
            "renderer": "buglab.scenes",
            "format": config.output_format,
            "fps": config.fps,
            "frame_count": config.frame_count,
            "canvas_inches": [style.canvas_width, style.canvas_height],
            "dpi": 100,
            "canvas_pixels": [int(style.canvas_width * 100), int(style.canvas_height * 100)],
            "artifact_sha256": _sha256(output_path),
        },
        "rendering_launches_optimizer": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def export_manifest(
    configs: tuple[SceneConfig, ...], output_dir: Path, *, data: PlaybackData | None = None,
    source_root: Path = Path("."),
) -> list[tuple[Path, Path]]:
    playback_cache: dict[str, PlaybackData] = {}
    operation_cache: dict[str, list[GeneticOperation]] = {}

    def playback(path: str) -> PlaybackData:
        if path not in playback_cache:
            playback_cache[path] = load_playback_data(source_root / path)
        return playback_cache[path]

    exports = []
    for config in configs:
        primary = data if config.data_path is None else playback(config.data_path)
        operations = None
        paths: dict[str, str] = {}
        if config.data_path is not None:
            paths["primary"] = config.data_path
        if config.operation_log_path is not None:
            if config.operation_log_path not in operation_cache:
                operation_cache[config.operation_log_path] = load_operation_log(
                    source_root / config.operation_log_path
                )
            operations = operation_cache[config.operation_log_path]
            paths["operation_log"] = config.operation_log_path
        comparisons = None
        if config.comparison_data_paths is not None:
            comparisons = {method: playback(path)
                           for method, path in config.comparison_data_paths.items()}
            paths.update({f"comparison_{method}": path
                          for method, path in config.comparison_data_paths.items()})
        exports.append(export_scene(
            config, output_dir, data=primary, datasets=comparisons,
            operations=operations, source_paths=paths,
        ))
    return exports


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Export deterministic Bug Evolution Lab lecture scenes")
    parser.add_argument("--manifest", type=Path, default=Path("pilot_scenes.json"))
    parser.add_argument("--data", type=Path, help="recorded .npz playback cache")
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot_scenes"))
    args = parser.parse_args(argv)
    configs = load_scene_configs(args.manifest)
    data = load_playback_data(args.data) if args.data is not None else None
    for movie, metadata in export_manifest(
        configs, args.output, data=data, source_root=args.manifest.parent
    ):
        print(f"scene: {movie.resolve()}")
        print(f"metadata: {metadata.resolve()}")


if __name__ == "__main__":
    main()
