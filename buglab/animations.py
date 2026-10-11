"""Headless animations built only from recorded playback data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from matplotlib import animation, pyplot as plt
import numpy as np

from .playback import PlaybackData
from .visualization import field


METHOD_COLORS = {"ga": "#168579", "random": "#d17a22", "local": "#3973ac"}
METHOD_TITLES = {"ga": "Genetic algorithm", "random": "Random search", "local": "Local search"}


@dataclass(frozen=True)
class AnimationFrame:
    """One synchronized animation coordinate."""

    snapshot_index: int
    episode_step: int


def frame_schedule(data: PlaybackData, episode_frames: int = 10) -> tuple[AnimationFrame, ...]:
    """Separate optimizer iteration from within-episode beetle time."""
    if episode_frames < 2:
        raise ValueError("episode_frames must be at least 2")
    steps = np.linspace(0, data.world.steps, episode_frames, dtype=int)
    return tuple(
        AnimationFrame(snapshot_index, int(step))
        for snapshot_index in range(len(data.snapshots))
        for step in steps
    )


def trajectory_prefix(positions: np.ndarray, episode_step: int) -> np.ndarray:
    """Return a physical prefix, holding terminated beetles at their last position."""
    final_index = min(max(episode_step, 0), len(positions) - 1)
    return positions[: final_index + 1]


def _candidate_indices(data: PlaybackData, snapshot_index: int, limit: int) -> np.ndarray:
    snapshot = data.snapshot(snapshot_index)
    if data.config.method == "local":
        # Always retain incumbent candidate zero, then strongest neighbors.
        ranked = [int(index) for index in data.ranked_candidates(snapshot_index) if index != 0]
        return np.asarray(([0] + ranked)[:limit], dtype=int)
    return data.ranked_candidates(snapshot_index, limit)


def _preload(data: PlaybackData, limit: int) -> dict[tuple[int, int], np.ndarray]:
    """Replay each needed genome once before rendering any frame."""
    paths: dict[tuple[int, int], np.ndarray] = {}
    for snapshot_index in range(len(data.snapshots)):
        for candidate_index in _candidate_indices(data, snapshot_index, limit):
            paths[(snapshot_index, int(candidate_index))] = data.candidate_rollout(
                snapshot_index, int(candidate_index)
            ).positions
    return paths


def _status_text(data: PlaybackData, snapshot_index: int, episode_step: int) -> str:
    snapshot = data.snapshot(snapshot_index)
    semantics = data.candidate_semantics(snapshot_index)
    if data.config.method == "random":
        detail = "independent fresh samples"
    elif data.config.method == "ga":
        detail = "evaluated population; selection occurs between generations"
    elif snapshot_index == 0:
        detail = "initial candidates establish the incumbent"
    else:
        verdict = "accepted" if semantics["accepted"] else "rejected"
        detail = f"best neighbor {verdict}; incumbent shown in blue"
    success = "yes" if data.best_rollout(snapshot_index).reached else "no"
    return (
        f"{data.iteration_kind} {snapshot.generation} · episode step "
        f"{min(episode_step, data.world.steps)}/{data.world.steps}\n"
        f"{snapshot.evaluations} evaluations · best-so-far success: {success} · "
        f"fitness {snapshot.best_fitness:.2f}\n{detail}"
    )


def export_method_gif(
    data: PlaybackData,
    path: Path,
    *,
    episode_frames: int = 10,
    fps: int = 8,
    max_candidates: int = 10,
    show_checkpoint_markers: bool = False,
) -> None:
    """Animate real candidates using method-specific, scientifically correct semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schedule = frame_schedule(data, episode_frames)
    paths = _preload(data, max_candidates)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="white")
    field(ax, data.world, show_checkpoints=show_checkpoint_markers)
    tracks = [ax.plot([], [], color=METHOD_COLORS[data.config.method], alpha=.25, lw=1.2)[0]
              for _ in range(max_candidates)]
    dots = [ax.plot([], [], marker="o", ms=5, color=METHOD_COLORS[data.config.method])[0]
            for _ in range(max_candidates)]
    best_line, = ax.plot([], [], color="#20262d", lw=2.5, ls="--", label="Best so far")
    title = ax.set_title("", fontsize=14)
    ax.legend(loc="upper left", framealpha=.95)

    def draw(frame_index: int):
        coordinate = schedule[frame_index]
        indices = _candidate_indices(data, coordinate.snapshot_index, max_candidates)
        semantics = data.candidate_semantics(coordinate.snapshot_index)
        for slot, (line, dot) in enumerate(zip(tracks, dots)):
            if slot >= len(indices):
                line.set_data([], []); dot.set_data([], [])
                continue
            candidate_index = int(indices[slot])
            prefix = trajectory_prefix(
                paths[(coordinate.snapshot_index, candidate_index)], coordinate.episode_step
            )
            if data.config.method == "local" and coordinate.snapshot_index > 0:
                color = "#24558c" if candidate_index == 0 else "#94a6bd"
                if semantics["accepted"] and candidate_index == semantics["proposed_candidate_index"]:
                    color = "#2b8c5a"
                line.set_color(color); dot.set_color(color)
                line.set_alpha(.85 if candidate_index == 0 else .28)
            else:
                line.set_color(METHOD_COLORS[data.config.method])
                dot.set_color(METHOD_COLORS[data.config.method])
                line.set_alpha(.25)
            line.set_data(prefix[:, 0], prefix[:, 1])
            dot.set_data([prefix[-1, 0]], [prefix[-1, 1]])
        best_prefix = trajectory_prefix(
            data.best_rollout(coordinate.snapshot_index).positions, coordinate.episode_step
        )
        best_line.set_data(best_prefix[:, 0], best_prefix[:, 1])
        title.set_text(
            f"{METHOD_TITLES[data.config.method]} · {data.config.map_name} · "
            f"{data.config.fitness_name}\n{_status_text(data, coordinate.snapshot_index, coordinate.episode_step)}"
        )
        return (*tracks, *dots, best_line, title)

    movie = animation.FuncAnimation(fig, draw, frames=len(schedule), interval=1000 / fps,
                                    blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def validate_matched_playback(datasets: Mapping[str, PlaybackData]) -> None:
    """Reject misleading comparisons before rendering them."""
    required = {"ga", "random", "local"}
    if set(datasets) != required:
        raise ValueError(f"Matched comparison requires {sorted(required)}")
    values = tuple(datasets.values())
    reference = values[0]
    for data in values:
        if data.config.method not in required or datasets[data.config.method] is not data:
            raise ValueError("Dataset keys must match optimizer methods")
        shared = (
            data.config.map_name, data.config.fitness_name, data.config.seed,
            data.config.population, data.config.generations,
            data.world.steps, data.world.step_length, data.world.max_turn,
        )
        expected = (
            reference.config.map_name, reference.config.fitness_name, reference.config.seed,
            reference.config.population, reference.config.generations,
            reference.world.steps, reference.world.step_length, reference.world.max_turn,
        )
        if shared != expected:
            raise ValueError("Comparison runs must share map, fitness, seed, budget, and physics")
        if tuple(s.evaluations for s in data.snapshots) != tuple(
            s.evaluations for s in reference.snapshots
        ):
            raise ValueError("Comparison snapshots must share evaluation coordinates")


def export_comparison_gif(
    datasets: Mapping[str, PlaybackData],
    path: Path,
    *,
    episode_frames: int = 8,
    fps: int = 8,
    max_candidates: int = 8,
    show_checkpoint_markers: bool = False,
) -> None:
    """Export a synchronized three-panel comparison from cached snapshots."""
    validate_matched_playback(datasets)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [datasets[method] for method in ("ga", "random", "local")]
    schedule = frame_schedule(ordered[0], episode_frames)
    preloaded = {data.config.method: _preload(data, max_candidates) for data in ordered}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), facecolor="white", sharex=True, sharey=True)
    artists = {}
    for ax, data in zip(axes, ordered):
        field(ax, data.world, show_checkpoints=show_checkpoint_markers)
        tracks = [ax.plot([], [], color=METHOD_COLORS[data.config.method], alpha=.24, lw=1)[0]
                  for _ in range(max_candidates)]
        dots = [ax.plot([], [], marker="o", ms=4, color=METHOD_COLORS[data.config.method])[0]
                for _ in range(max_candidates)]
        best, = ax.plot([], [], color="#20262d", lw=2.2, ls="--")
        title = ax.set_title("", fontsize=11)
        artists[data.config.method] = (tracks, dots, best, title)
    reference = ordered[0]
    fig.suptitle(
        f"Matched optimizer comparison · map={reference.config.map_name} · "
        f"fitness={reference.config.fitness_name} · seed={reference.config.seed} · "
        f"budget={reference.config.generations * reference.config.population} evaluations",
        fontsize=16,
    )

    def draw(frame_index: int):
        coordinate = schedule[frame_index]
        changed = []
        for data in ordered:
            tracks, dots, best, title = artists[data.config.method]
            indices = _candidate_indices(data, coordinate.snapshot_index, max_candidates)
            for slot, (line, dot) in enumerate(zip(tracks, dots)):
                if slot >= len(indices):
                    line.set_data([], []); dot.set_data([], []); continue
                candidate_index = int(indices[slot])
                prefix = trajectory_prefix(
                    preloaded[data.config.method][(coordinate.snapshot_index, candidate_index)],
                    coordinate.episode_step,
                )
                line.set_data(prefix[:, 0], prefix[:, 1])
                dot.set_data([prefix[-1, 0]], [prefix[-1, 1]])
            best_prefix = trajectory_prefix(
                data.best_rollout(coordinate.snapshot_index).positions, coordinate.episode_step
            )
            best.set_data(best_prefix[:, 0], best_prefix[:, 1])
            snapshot = data.snapshot(coordinate.snapshot_index)
            success = "yes" if data.best_rollout(coordinate.snapshot_index).reached else "no"
            title.set_text(
                f"{METHOD_TITLES[data.config.method]}\n{data.iteration_kind} "
                f"{snapshot.generation} · episode step {coordinate.episode_step}/{data.world.steps}\n"
                f"evals {snapshot.evaluations} · success {success} · best {snapshot.best_fitness:.1f}"
            )
            changed.extend((*tracks, *dots, best, title))
        return changed

    movie = animation.FuncAnimation(fig, draw, frames=len(schedule), interval=1000 / fps,
                                    blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def export_comparison_png(
    datasets: Mapping[str, PlaybackData], path: Path, *, max_candidates: int = 10,
    show_checkpoint_markers: bool = False,
) -> None:
    """Render the final matched snapshot as a 16:9 slide."""
    validate_matched_playback(datasets)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [datasets[method] for method in ("ga", "random", "local")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), facecolor="white", sharex=True, sharey=True)
    for ax, data in zip(axes, ordered):
        index = len(data.snapshots) - 1
        snapshot = data.snapshot(index)
        field(ax, data.world, show_checkpoints=show_checkpoint_markers)
        for candidate_index in _candidate_indices(data, index, max_candidates):
            rollout = data.candidate_rollout(index, int(candidate_index))
            ax.plot(rollout.positions[:, 0], rollout.positions[:, 1],
                    color=METHOD_COLORS[data.config.method], alpha=.22, lw=1)
        best = data.best_rollout(index)
        ax.plot(best.positions[:, 0], best.positions[:, 1], color="#20262d", lw=2.6,
                ls="--", label="Best so far")
        ax.set_title(
            f"{METHOD_TITLES[data.config.method]}\n{snapshot.evaluations} evaluations · "
            f"success {'yes' if best.reached else 'no'} · best {snapshot.best_fitness:.1f}",
            fontsize=12,
        )
        ax.legend(loc="upper left", fontsize=9)
    reference = ordered[0]
    fig.suptitle(
        f"Matched comparison · {reference.config.map_name} · {reference.config.fitness_name} · "
        f"seed {reference.config.seed}", fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
