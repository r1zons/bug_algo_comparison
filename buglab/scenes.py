"""Declarative, deterministic lecture scenes rendered from recorded data."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from matplotlib import animation, patches, pyplot as plt
import numpy as np

from .operator_story import SyntheticOperation, synthetic_operation
from .playback import PlaybackData, load_playback_data
from .visualization import PARENT_A_COLOR, PARENT_B_COLOR, MUTATION_COLOR, field


SCENE_KINDS = ("single_beetle", "initial_population", "synthetic_crossover")
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

    @property
    def frame_count(self) -> int:
        return max(2, int(round(self.duration_seconds * self.fps)))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SceneConfig":
        return cls(**dict(values))


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
    steps = np.linspace(0, data.world.steps, config.frame_count, dtype=int)
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height),
                        facecolor=style.background, constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.65, 1.0))
    map_ax = figure.add_subplot(grid[0, 0])
    genes_ax = figure.add_subplot(grid[0, 1])
    field(map_ax, data.world, show_checkpoints=config.show_checkpoint_markers)
    route, = map_ax.plot([], [], color=style.trajectory, lw=3, label="Executed trajectory")
    beetle, = map_ax.plot([], [], marker="o", ms=9, color=style.highlight, zorder=10)
    collision, = map_ax.plot([], [], marker="x", ms=10, mew=2, color=style.collision,
                             visible=False, label="Blocked move")
    map_ax.legend(loc="upper left", framealpha=.95)
    degrees = np.rad2deg(genome)
    genes_ax.axhline(0, color="#697988", lw=1)
    genes_ax.plot(np.arange(len(genome)), degrees, color="#aeb8c1", lw=1.3)
    current_gene, = genes_ax.plot([], [], marker="o", ms=8, color=style.highlight)
    cursor = genes_ax.axvline(0, color=style.highlight, lw=2)
    genes_ax.set_xlim(-1, len(genome)); genes_ax.set_ylim(-27, 27)
    genes_ax.set_xlabel("gene index / episode step")
    genes_ax.set_ylabel("relative turn angle (degrees)")
    genes_ax.set_title("Genome: a sequence of steering commands", fontsize=13)
    genes_ax.grid(alpha=.18)
    status = genes_ax.text(.02, .96, "", transform=genes_ax.transAxes, va="top", fontsize=11,
                           bbox={"boxstyle": "round,pad=.5", "fc": "white", "ec": "#c7cdd3"})
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    figure.text(.5, .01,
                "Actual deterministic simulation · fixed-length movement · relative turns stored in radians",
                ha="center", fontsize=10)

    def draw(frame_index: int):
        requested_step = int(steps[frame_index])
        position_index = min(requested_step, len(rollout.positions) - 1)
        prefix = rollout.positions[: position_index + 1]
        route.set_data(prefix[:, 0], prefix[:, 1])
        beetle.set_data([prefix[-1, 0]], [prefix[-1, 1]])
        gene_index = 0 if position_index == 0 else min(position_index - 1, len(genome) - 1)
        current_gene.set_data([gene_index], [degrees[gene_index]])
        cursor.set_xdata([gene_index, gene_index])
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
            f"playback step {requested_step:02d}/{data.world.steps}\n"
            f"{execution_text}\nlast/next turn {degrees[gene_index]:+.1f}°\n"
            f"blocked move: {'yes' if blocked else 'no'}\n"
            f"goal reached: {'yes' if reached else 'no'}"
        )
        map_ax.set_title(f"Recorded position at playback step {requested_step}", fontsize=13)
        return route, beetle, collision, current_gene, cursor, status

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
    steps = np.linspace(0, data.world.steps, config.frame_count, dtype=int)
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height),
                        facecolor=style.background)
    grid = figure.add_gridspec(1, 2, width_ratios=(2.2, .8))
    map_ax = figure.add_subplot(grid[0, 0]); panel = figure.add_subplot(grid[0, 1])
    field(map_ax, data.world, show_checkpoints=config.show_checkpoint_markers)
    map_ax.set_xlabel("x coordinate")
    lines = [map_ax.plot([], [], color=style.population, alpha=.3, lw=1.3)[0]
             for _ in candidate_indices]
    dots = [map_ax.plot([], [], marker="o", color=style.trajectory, alpha=.85, ms=5)[0]
            for _ in candidate_indices]
    panel.axis("off")
    panel.text(.04, .92, "One shared world", fontsize=15, fontweight="bold")
    panel.text(.04, .82,
               "Each line is one independently\nsampled turn-angle genome.\n\n"
               "Every candidate receives:\n"
               "• identical physics\n• identical episode horizon\n• identical reward policy",
               fontsize=12, va="top", linespacing=1.45)
    status = panel.text(.04, .38, "", fontsize=12, va="top",
                        bbox={"boxstyle": "round,pad=.6", "fc": "#f7f9fa", "ec": "#c7cdd3"})
    figure.suptitle(config.title, fontsize=18, fontweight="bold")
    figure.subplots_adjust(left=.055, right=.975, bottom=.12, top=.87, wspace=.16)
    figure.text(.5, .025,
                "Actual recorded random-search batch · trajectories are not inherited offspring",
                ha="center", fontsize=10)

    def draw(frame_index: int):
        requested_step = int(steps[frame_index])
        active = 0; successes = 0
        for rollout, line, dot in zip(rollouts, lines, dots):
            position_index = min(requested_step, len(rollout.positions) - 1)
            prefix = rollout.positions[: position_index + 1]
            line.set_data(prefix[:, 0], prefix[:, 1])
            dot.set_data([prefix[-1, 0]], [prefix[-1, 1]])
            active += int(position_index < len(rollout.positions) - 1)
            successes += int(rollout.reached and position_index == len(rollout.positions) - 1)
        status.set_text(
            f"batch 0\nepisode step {requested_step}/{data.world.steps}\n"
            f"candidates shown {count}\nstill moving {active}\nphysical successes {successes}"
        )
        map_ax.set_title("Different genomes produce different paths", fontsize=14)
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


def export_scene(
    config: SceneConfig,
    output_dir: Path,
    *,
    data: PlaybackData | None = None,
    style: LectureStyle = LectureStyle(),
) -> tuple[Path, Path]:
    """Render one scene. This function never creates or runs an optimizer."""
    _apply_style(style)
    if config.output_format not in available_export_formats():
        raise RuntimeError(f"Export format {config.output_format!r} is unavailable")
    if config.kind == "synthetic_crossover":
        movie, figure, source = _synthetic_movie(config, synthetic_operation(), style)
        data_metadata = None
    else:
        if data is None:
            raise ValueError(f"Scene {config.kind!r} requires recorded PlaybackData")
        if config.kind == "single_beetle":
            movie, figure, source = _single_beetle_movie(config, data, style)
        else:
            movie, figure, source = _population_movie(config, data, style)
        data_metadata = asdict(data.config)
    output_path = output_dir / f"{config.name}.{config.output_format}"
    metadata_path = output_dir / f"{config.name}.json"
    _save_movie(movie, figure, output_path, config)
    metadata = {
        "schema_version": 1,
        "scene": asdict(config),
        "style": asdict(style),
        "source_playback": data_metadata,
        "source_details": source,
        "rendering_launches_optimizer": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def export_manifest(
    configs: tuple[SceneConfig, ...], output_dir: Path, *, data: PlaybackData | None = None
) -> list[tuple[Path, Path]]:
    return [export_scene(config, output_dir, data=data) for config in configs]


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Export deterministic Bug Evolution Lab lecture scenes")
    parser.add_argument("--manifest", type=Path, default=Path("pilot_scenes.json"))
    parser.add_argument("--data", type=Path, help="recorded .npz playback cache")
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot_scenes"))
    args = parser.parse_args(argv)
    configs = load_scene_configs(args.manifest)
    data = load_playback_data(args.data) if args.data is not None else None
    for movie, metadata in export_manifest(configs, args.output, data=data):
        print(f"scene: {movie.resolve()}")
        print(f"metadata: {metadata.resolve()}")


if __name__ == "__main__":
    main()
