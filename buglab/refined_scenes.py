"""Focused Act I method-mechanics and rectangular-spiral lecture exports."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from matplotlib import animation, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from .animations import METHOD_COLORS, METHOD_TITLES, validate_matched_playback
from .maps import rectangular_spiral_distance_barrier
from .operator_story import GeneticOperation, load_operation_log
from .playback import PlaybackData, load_playback_data
from .reference_paths import rectangular_spiral_reference
from .scenes import (LectureStyle, SceneConfig, export_scene,
                     interpolated_trajectory_prefix)
from .visualization import MUTATION_COLOR, PARENT_A_COLOR, PARENT_B_COLOR, field


@dataclass(frozen=True)
class RefinedSceneConfig:
    name: str
    kind: str
    duration_seconds: float
    fps: int
    educational_message: str
    data_path: str | None = None
    operation_log_path: str | None = None
    comparison_data_paths: Mapping[str, str] | None = None
    snapshot_indices: tuple[int, ...] = ()
    max_candidates: int = 10

    @property
    def frame_count(self) -> int:
        return max(2, round(self.duration_seconds * self.fps))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RefinedSceneConfig":
        parsed = dict(values)
        parsed["snapshot_indices"] = tuple(parsed.get("snapshot_indices", ()))
        if parsed.get("comparison_data_paths") is not None:
            parsed["comparison_data_paths"] = dict(parsed["comparison_data_paths"])
        return cls(**parsed)


def load_refined_manifest(path: Path) -> tuple[RefinedSceneConfig, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = tuple(RefinedSceneConfig.from_mapping(item) for item in payload["scenes"])
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("Refined scene names must be unique")
    return configs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _genome_sha256(genome: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(genome, dtype=np.float64).tobytes()).hexdigest()


def _block_schedule(frame_count: int, blocks: int) -> list[tuple[int, float]]:
    base, extra = divmod(frame_count, blocks)
    result: list[tuple[int, float]] = []
    for block in range(blocks):
        count = base + int(block < extra)
        result.extend((block, index / max(1, count - 1)) for index in range(count))
    return result[:frame_count]


def _save_custom(
    config: RefinedSceneConfig,
    output_dir: Path,
    movie: animation.FuncAnimation,
    figure,
    draw: Callable[[int], Any],
    source_details: dict[str, Any],
    source_paths: Mapping[str, str],
    *,
    poster_frame: int | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{config.name}.gif"
    poster_path = output_dir / f"{config.name}_poster.png"
    metadata_path = output_dir / f"{config.name}.json"
    movie.save(gif_path, writer=animation.PillowWriter(fps=config.fps), dpi=100)
    draw(config.frame_count - 1 if poster_frame is None else poster_frame)
    figure.savefig(poster_path, dpi=150, facecolor="white")
    plt.close(figure)
    metadata = {
        "schema_version": 1,
        "scene": asdict(config),
        "source_data_paths": dict(source_paths),
        "source_data_sha256": {
            label: _sha256(Path(path)) for label, path in source_paths.items()
        },
        "source_details": source_details,
        "export": {
            "renderer": "buglab.refined_scenes",
            "fps": config.fps,
            "configured_frame_count": config.frame_count,
            "canvas_pixels": [1280, 720],
            "gif_sha256": _sha256(gif_path),
            "poster_sha256": _sha256(poster_path),
        },
        "rendering_launches_optimizer": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return gif_path, poster_path, metadata_path


def _random_movie(config: RefinedSceneConfig, data: PlaybackData, style: LectureStyle):
    indices = config.snapshot_indices or (0, 1, 2, 3)
    candidate = 0
    schedule = _block_schedule(config.frame_count, len(indices))
    rollouts = {index: data.candidate_rollout(index, candidate) for index in indices}
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    map_ax = figure.add_axes((.09, .17, .82, .69))
    genome_ax = figure.add_axes((.18, .075, .64, .045))
    figure.text(.5, .925, "Каждая попытка — заново сэмплированный полный геном",
                ha="center", va="center", fontsize=13, fontweight="bold")
    figure.text(.5, .025,
                "Начальный курс направлен к цели; малые относительные повороты дают плавные траектории.",
                ha="center", fontsize=11.5)

    def draw(frame_index: int):
        block, phase = schedule[frame_index]
        snapshot_index = indices[block]
        snapshot = data.snapshot(snapshot_index)
        rollout = rollouts[snapshot_index]
        map_ax.clear(); field(map_ax, data.world, show_checkpoints=False)
        prefix = interpolated_trajectory_prefix(
            rollout.positions, min(1.0, phase / .78) * (len(rollout.positions) - 1)
        )
        map_ax.plot(prefix[:, 0], prefix[:, 1], color=METHOD_COLORS["random"], lw=3,
                    label="новая независимая попытка")
        map_ax.plot(prefix[-1, 0], prefix[-1, 1], "o", color=METHOD_COLORS["random"], ms=8)
        if snapshot_index > 0:
            previous_best = data.best_rollout(snapshot_index - 1).positions
            map_ax.plot(previous_best[:, 0], previous_best[:, 1], color=style.highlight,
                        lw=2.2, ls="--", alpha=.85,
                        label="лучший ранее — только для отчёта")
        map_ax.legend(loc="upper left", fontsize=10, framealpha=.95)
        map_ax.text(.985, .04,
                    f"пакет {snapshot.generation} · кандидат {candidate}\n"
                    f"fitness {snapshot.candidate_scores[candidate]:.2f} · seed {data.config.seed}",
                    transform=map_ax.transAxes, ha="right", va="bottom", fontsize=11,
                    bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#c7cdd3"})
        genome_ax.clear()
        genome = snapshot.candidate_genomes[candidate]
        genome_ax.imshow(genome[np.newaxis, :], aspect="auto", cmap="coolwarm",
                         vmin=-data.world.max_turn, vmax=data.world.max_turn)
        genome_ax.set_yticks([]); genome_ax.set_xticks((0, len(genome) - 1), ("0", str(len(genome)-1)))
        genome_ax.set_title("90 заранее заданных команд поворота", fontsize=9, pad=2)
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000/config.fps, blit=False)
    details = {
        "method_semantics": (
            "Every batch is sampled independently from Normal(0, init_sigma), clipped to turn bounds; "
            "tell() retains no genome for generation. Best-so-far is reporting only."
        ),
        "initial_heading": "The simulator initially points directly from start to goal.",
        "init_sigma_radians": .11,
        "candidate_records": [
            {"snapshot_index": index, "candidate_index": candidate,
             "score": float(data.snapshot(index).candidate_scores[candidate]),
             "genome_sha256": _genome_sha256(data.snapshot(index).candidate_genomes[candidate])}
            for index in indices
        ],
    }
    return movie, figure, draw, details


def _local_movie(config: RefinedSceneConfig, data: PlaybackData, style: LectureStyle):
    indices = config.snapshot_indices or (3, 4, 5, 6)
    schedule = _block_schedule(config.frame_count, len(indices))
    paths = {
        (snapshot_index, candidate): data.candidate_rollout(snapshot_index, candidate).positions
        for snapshot_index in indices for candidate in range(data.config.population)
    }
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(3.2, 1), left=.055, right=.97,
                               bottom=.12, top=.88, wspace=.14)
    map_ax = figure.add_subplot(grid[0, 0]); score_ax = figure.add_subplot(grid[0, 1])
    figure.text(.5, .035,
                "Решение принимается между полными эпизодами — это не онлайн-навигация.",
                ha="center", fontsize=12, fontweight="bold")

    records = []
    for snapshot_index in indices:
        snapshot = data.snapshot(snapshot_index)
        semantics = data.candidate_semantics(snapshot_index)
        neighbor = int(semantics["best_neighbor_candidate_index"])
        records.append({
            "snapshot_index": snapshot_index,
            "incumbent_candidate_index": 0,
            "best_neighbor_candidate_index": neighbor,
            "incumbent_score": float(snapshot.candidate_scores[0]),
            "best_neighbor_score": float(snapshot.candidate_scores[neighbor]),
            "accepted": bool(semantics["accepted"]),
            "incumbent_genome_sha256": _genome_sha256(snapshot.candidate_genomes[0]),
            "neighbor_genome_sha256": _genome_sha256(snapshot.candidate_genomes[neighbor]),
        })

    def draw(frame_index: int):
        block, phase = schedule[frame_index]
        snapshot_index = indices[block]
        snapshot = data.snapshot(snapshot_index)
        semantics = data.candidate_semantics(snapshot_index)
        best_neighbor = int(semantics["best_neighbor_candidate_index"])
        accepted = bool(semantics["accepted"])
        map_ax.clear(); score_ax.clear(); field(map_ax, data.world, show_checkpoints=False)
        incumbent_path = paths[(snapshot_index, 0)]
        incumbent_fraction = min(1.0, phase / .2)
        incumbent = interpolated_trajectory_prefix(
            incumbent_path, incumbent_fraction * (len(incumbent_path) - 1)
        )
        map_ax.plot(incumbent[:, 0], incumbent[:, 1], color="#2268a8", lw=3.2,
                    label="текущий инкумбент")
        ranked_neighbors = [int(index) for index in np.argsort(snapshot.candidate_scores[1:])[::-1] + 1]
        shown = ranked_neighbors[:min(config.max_candidates, len(ranked_neighbors))]
        neighbor_fraction = np.clip((phase - .2) / .5, 0, 1)
        for candidate in shown:
            path = paths[(snapshot_index, candidate)]
            prefix = interpolated_trajectory_prefix(
                path, neighbor_fraction * (len(path) - 1)
            )
            color = style.highlight if candidate == best_neighbor and accepted else (
                style.collision if candidate == best_neighbor else "#89a9c2"
            )
            map_ax.plot(prefix[:, 0], prefix[:, 1], color=color,
                        alpha=.95 if candidate == best_neighbor else .28,
                        lw=2.8 if candidate == best_neighbor else 1.1)
        if phase >= .72:
            verdict = "УЛУЧШЕНИЕ ПРИНЯТО" if accepted else "ОКРЕСТНОСТЬ ОТКЛОНЕНА"
            verdict_color = style.highlight if accepted else style.collision
            map_ax.text(.5, .96, verdict, transform=map_ax.transAxes, ha="center", va="top",
                        fontsize=13, fontweight="bold", color=verdict_color,
                        bbox={"boxstyle": "round,pad=.35", "fc": "white",
                              "ec": verdict_color})
        else:
            map_ax.text(.5, .96,
                        "Сначала инкумбент, затем полные траектории мутированных соседей",
                        transform=map_ax.transAxes, ha="center", va="top", fontsize=11.5,
                        fontweight="bold", bbox={"boxstyle": "round,pad=.3", "fc": "white",
                                                  "ec": "#c7cdd3"})
        map_ax.legend(handles=[
            Line2D([0], [0], color="#2268a8", lw=3, label="инкумбент"),
            Line2D([0], [0], color="#89a9c2", lw=2, label="мутированные соседи"),
            Line2D([0], [0], color=style.highlight if accepted else style.collision, lw=3,
                   label="лучший сосед"),
        ], loc="upper left", fontsize=9.5)
        score_ax.set_title("Fitness после эпизода", fontsize=12, fontweight="bold")
        score_ax.axvline(snapshot.candidate_scores[0], color="#2268a8", lw=2,
                         label="инкумбент")
        score_ax.scatter(snapshot.candidate_scores[1:], np.arange(1, data.config.population),
                         color="#89a9c2", s=28, alpha=.7)
        score_ax.scatter([snapshot.candidate_scores[best_neighbor]], [best_neighbor],
                         color=style.highlight if accepted else style.collision, s=85, zorder=4)
        score_ax.set_yticks([]); score_ax.set_xlabel("выше — лучше", fontsize=10)
        score_ax.grid(axis="x", alpha=.2)
        score_ax.text(.97, .04,
                      f"пакет {snapshot.generation}\nseed {data.config.seed}\n"
                      f"инкумбент {snapshot.candidate_scores[0]:.2f}\n"
                      f"лучший сосед {snapshot.candidate_scores[best_neighbor]:.2f}",
                      transform=score_ax.transAxes, ha="right", va="bottom", fontsize=10)
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000/config.fps, blit=False)
    details = {
        "method_semantics": (
            "Candidate 0 is the unchanged incumbent; candidates 1..N are mutated complete genomes. "
            "Acceptance occurs in tell() only after complete episode scores are returned."
        ),
        "online_navigation": False,
        "batch_records": records,
    }
    return movie, figure, draw, details


def _ga_movie(config: RefinedSceneConfig, data: PlaybackData,
              operations: list[GeneticOperation], style: LectureStyle):
    event = next(operation for operation in operations
                 if operation.kind == "offspring" and operation.generation == 1
                 and operation.mutated_indices)
    before_index = data.snapshot_index(event.generation - 1)
    after_index = data.snapshot_index(event.generation)
    before = data.snapshot(before_index); after = data.snapshot(after_index)
    population_paths = [data.candidate_rollout(before_index, index).positions
                        for index in range(min(config.max_candidates, data.config.population))]
    parent_paths = (data.candidate_rollout(before_index, event.parent_a_id).positions,
                    data.candidate_rollout(before_index, int(event.parent_b_id)).positions)
    child_path = data.candidate_rollout(after_index, event.candidate_index).positions
    progress = np.linspace(0, 1, config.frame_count)
    figure, ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    figure.subplots_adjust(left=.08, right=.92, bottom=.12, top=.88)
    figure.text(.5, .035,
                "Популяция → отбор родителей → кроссовер и мутация → оценка нового кандидата",
                ha="center", fontsize=12, fontweight="bold")

    def draw(frame_index: int):
        phase = float(progress[frame_index]); ax.clear(); field(ax, data.world, show_checkpoints=False)
        if phase < .32:
            fraction = min(1.0, phase / .25)
            for path in population_paths:
                prefix = interpolated_trajectory_prefix(path, fraction * (len(path)-1))
                ax.plot(prefix[:, 0], prefix[:, 1], color=METHOD_COLORS["ga"], alpha=.28, lw=1.3)
            label = "1 · оценённая популяция"
        elif phase < .55:
            for path, color, label_name in zip(parent_paths,
                                                (PARENT_A_COLOR, PARENT_B_COLOR),
                                                ("родитель A", "родитель B")):
                ax.plot(path[:, 0], path[:, 1], color=color, lw=3, label=label_name)
            ax.legend(loc="upper left", fontsize=10)
            label = "2 · выбранная пара родителей"
        elif phase < .68:
            for path, color in zip(parent_paths, (PARENT_A_COLOR, PARENT_B_COLOR)):
                ax.plot(path[:, 0], path[:, 1], color=color, lw=2.5, alpha=.8)
            ax.text(.5, .48,
                    f"срез: ген {event.cut}   ·   изменено генов: {len(event.mutated_indices)}",
                    transform=ax.transAxes, ha="center", fontsize=16, fontweight="bold",
                    bbox={"boxstyle": "round,pad=.6", "fc": "white", "ec": MUTATION_COLOR})
            label = "3 · команды объединены и мутированы"
        else:
            fraction = min(1.0, (phase - .68) / .24)
            for candidate in range(min(config.max_candidates, data.config.population)):
                path = data.candidate_rollout(after_index, candidate).positions
                prefix = interpolated_trajectory_prefix(path, fraction * (len(path)-1))
                ax.plot(prefix[:, 0], prefix[:, 1], color=METHOD_COLORS["ga"], alpha=.18, lw=1)
            child = interpolated_trajectory_prefix(child_path, fraction * (len(child_path)-1))
            ax.plot(child[:, 0], child[:, 1], color=MUTATION_COLOR, lw=3.2,
                    label="записанный новый кандидат")
            ax.legend(loc="upper left", fontsize=10)
            label = "4 · новый кандидат проходит полный эпизод"
        ax.text(.5, .97, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=13, fontweight="bold",
                bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#c7cdd3"})
        ax.text(.985, .04, f"seed {data.config.seed} · поколение 0 → 1",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5)
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000/config.fps, blit=False)
    details = {
        "method_semantics": "population, tournament selection, one-point crossover, mutation, elitism",
        "operation_id": event.operation_id,
        "generation": event.generation,
        "parent_ids": [event.parent_a_id, event.parent_b_id],
        "parent_scores": [event.parent_a_score, event.parent_b_score],
        "cut": event.cut,
        "mutated_indices": event.mutated_indices,
        "child_candidate_index": event.candidate_index,
        "child_score": event.child_score,
        "child_genome_sha256": _genome_sha256(event.child),
    }
    return movie, figure, draw, details


def _barrier_movie(config: RefinedSceneConfig, style: LectureStyle):
    world, genome, trace = rectangular_spiral_reference()
    certificate = rectangular_spiral_distance_barrier(world.radius)
    distances = np.linalg.norm(trace.positions - np.asarray(world.goal), axis=1)
    running_minimum = np.minimum.accumulate(distances)
    progress = np.linspace(0, 1, config.frame_count)
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(2.8, 1), left=.055, right=.97,
                               bottom=.13, top=.9, wspace=.16)
    map_ax = figure.add_subplot(grid[0, 0]); distance_ax = figure.add_subplot(grid[0, 1])

    def draw(frame_index: int):
        phase = float(progress[frame_index]); map_ax.clear(); distance_ax.clear()
        field(map_ax, world, show_checkpoints=False)
        map_ax.axvline(certificate.separator_x, color=MUTATION_COLOR, lw=2, ls="--",
                       alpha=.75, label="обязательный обход справа")
        map_ax.add_patch(plt.Circle(world.goal, certificate.starting_distance, fill=False,
                                   ec="#d17a22", ls=":", lw=2, alpha=.75))
        fraction = np.clip((phase - .15) / .7, 0, 1)
        position = fraction * (len(trace.positions) - 1)
        prefix = interpolated_trajectory_prefix(trace.positions, position)
        map_ax.plot(prefix[:, 0], prefix[:, 1], color="#526b7f", lw=3,
                    label="физически исполнимый маршрут")
        prefix_distances = np.linalg.norm(prefix - np.asarray(world.goal), axis=1)
        outside = prefix_distances > certificate.starting_distance + 1e-9
        if np.any(outside):
            map_ax.plot(prefix[outside, 0], prefix[outside, 1], color=MUTATION_COLOR, lw=4)
        map_ax.plot(prefix[-1, 0], prefix[-1, 1], "o", color="#526b7f", ms=8)
        map_ax.legend(loc="lower right", fontsize=9.5, framealpha=.95)
        map_ax.text(.5, .97,
                    f"Любой допустимый путь должен увеличить расстояние минимум на Δ = {certificate.delta:.1f}",
                    transform=map_ax.transAxes, ha="center", va="top", fontsize=12,
                    fontweight="bold", bbox={"boxstyle": "round,pad=.3", "fc": "white",
                                              "ec": MUTATION_COLOR})
        shown = max(1, int(round(position)) + 1)
        distance_ax.plot(np.arange(shown), distances[:shown], color=MUTATION_COLOR, lw=2.5,
                         label="текущее расстояние")
        distance_ax.plot(np.arange(shown), running_minimum[:shown], color="#d17a22", lw=2.5,
                         label="минимум до этого шага")
        distance_ax.axhline(certificate.starting_distance, color="#8a96a3", ls=":", lw=1.5)
        distance_ax.set_title("Расстояние до цели", fontsize=12, fontweight="bold")
        distance_ax.set_xlabel("шаг траектории"); distance_ax.set_ylabel("расстояние")
        distance_ax.set_xlim(0, len(trace.positions)-1)
        distance_ax.set_ylim(0, max(distances) * 1.08)
        distance_ax.grid(alpha=.2); distance_ax.legend(loc="upper right", fontsize=8.5)
        distance_ax.text(.5, .03,
                         "final_euclidean ухудшается при движении прочь;\n"
                         "euclidean хранит прошлый минимум и выходит на плато.",
                         transform=distance_ax.transAxes, ha="center", va="bottom", fontsize=9.2,
                         bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "#c7cdd3"})
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000/config.fps, blit=False)
    details = {
        "source_type": "analytic geometry plus deterministic physics-valid reference",
        "distance_barrier": asdict(certificate),
        "reference_genome_sha256": _genome_sha256(genome),
        "reference_reached_goal": trace.reached,
        "reference_reached_at": trace.reached_at,
        "reference_collisions": trace.collisions,
        "reference_is_optimizer_input": False,
        "maximum_reference_distance": float(distances.max()),
        "fitness_distinction": {
            "euclidean": "negative minimum distance over the complete episode",
            "final_euclidean": "negative distance at the final episode position",
        },
    }
    return movie, figure, draw, details


def _rectangular_optimizer_movie(config: RefinedSceneConfig, data: PlaybackData,
                                  style: LectureStyle):
    indices = config.snapshot_indices or tuple(range(len(data.snapshots)))
    schedule = _block_schedule(config.frame_count, len(indices))
    count = min(config.max_candidates, data.config.population)
    paths = {(snapshot_index, candidate): data.candidate_rollout(snapshot_index, candidate).positions
             for snapshot_index in indices for candidate in range(count)}
    figure, ax = plt.subplots(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    figure.subplots_adjust(left=.07, right=.93, bottom=.12, top=.89)

    def draw(frame_index: int):
        block, phase = schedule[frame_index]
        snapshot_index = indices[block]; snapshot = data.snapshot(snapshot_index)
        ax.clear(); field(ax, data.world, show_checkpoints=False)
        fraction = min(1.0, phase / .8)
        for candidate in range(count):
            path = paths[(snapshot_index, candidate)]
            prefix = interpolated_trajectory_prefix(path, fraction * (len(path)-1))
            ax.plot(prefix[:, 0], prefix[:, 1], color="#d17a22", alpha=.2, lw=1.1)
        selected = data.selected_candidate(snapshot_index)
        selected_rollout = data.candidate_rollout(snapshot_index, selected)
        selected_prefix = interpolated_trajectory_prefix(
            selected_rollout.positions, fraction * (len(selected_rollout.positions)-1)
        )
        ax.plot(selected_prefix[:, 0], selected_prefix[:, 1], color="#d17a22", lw=3,
                label="лучший кандидат текущего поколения")
        if phase >= .8:
            distances = np.linalg.norm(selected_rollout.positions - np.asarray(data.world.goal), axis=1)
            closest = int(np.argmin(distances))
            ax.plot(selected_rollout.positions[closest, 0], selected_rollout.positions[closest, 1],
                    "o", mfc="white", mec=MUTATION_COLOR, mew=2, ms=9,
                    label="минимальное расстояние в эпизоде")
        ax.legend(loc="lower right", fontsize=9.5)
        success = any(bool(row["goal_reached"]) for row in data.history[:snapshot.generation+1])
        ax.text(.985, .90,
                f"поколение {snapshot.generation} · {snapshot.evaluations} оценок · seed {data.config.seed}\n"
                f"физический успех: {'да' if success else 'нет'}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#c7cdd3"})
        ax.text(.5, .97, r"Реальный запуск GA · fitness = $-\min_t$ расстояние",
                transform=ax.transAxes, ha="center", va="top", fontsize=12.5,
                fontweight="bold", bbox={"boxstyle": "round,pad=.3", "fc": "white",
                                          "ec": "#c7cdd3"})
        return ()

    movie = animation.FuncAnimation(figure, draw, frames=config.frame_count,
                                    interval=1000/config.fps, blit=False)
    details = {
        "method": data.config.method,
        "fitness": data.config.fitness_name,
        "fitness_definition": "negative minimum Euclidean distance over the full episode",
        "seed": data.config.seed,
        "evaluation_budget": data.config.generations * data.config.population,
        "run_goal_reached": any(bool(row["goal_reached"]) for row in data.history),
        "checkpoint_markers_shown": False,
        "reference_route_shown": False,
        "reference_route_available_to_optimizer": False,
    }
    return movie, figure, draw, details


def _poster_from_gif(gif_path: Path, poster_path: Path) -> None:
    with Image.open(gif_path) as image:
        image.seek(image.n_frames - 1)
        image.convert("RGB").save(poster_path)


def _comparison_export(config: RefinedSceneConfig, output_dir: Path,
                       datasets: Mapping[str, PlaybackData], source_paths: Mapping[str, str]):
    validate_matched_playback(datasets)
    scene = SceneConfig(
        name=config.name, kind="method_comparison", title="Три способа создавать кандидатов",
        duration_seconds=config.duration_seconds, fps=config.fps,
        max_candidates=config.max_candidates,
        comparison_data_paths=config.comparison_data_paths,
        educational_message=config.educational_message,
    )
    gif_path, metadata_path = export_scene(
        scene, output_dir, datasets=datasets, source_paths=source_paths
    )
    poster_path = output_dir / f"{config.name}_poster.png"
    _poster_from_gif(gif_path, poster_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_details"]["interpretation_scope"] = (
        "one matched seed illustrates candidate-generation semantics, not optimizer superiority"
    )
    metadata["export"]["poster_sha256"] = _sha256(poster_path)
    metadata["export"]["gif_sha256"] = _sha256(gif_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return gif_path, poster_path, metadata_path


def _contact_sheet(configs: tuple[RefinedSceneConfig, ...], output_dir: Path) -> tuple[Path, Path]:
    figure, axes = plt.subplots(len(configs), 4, figsize=(16, 3.1*len(configs)), facecolor="white")
    axes = np.atleast_2d(axes)
    for column, title in enumerate(("First", "Middle", "Final", "Message")):
        axes[0, column].set_title(title, fontsize=15, fontweight="bold")
    records = {}
    for row, config in enumerate(configs):
        path = output_dir / f"{config.name}.gif"
        with Image.open(path) as image:
            indices = (0, image.n_frames//2, image.n_frames-1)
            records[config.name] = {"frames": image.n_frames, "indices": indices}
            for column, index in enumerate(indices):
                image.seek(index)
                axes[row, column].imshow(image.convert("RGB"))
                axes[row, column].axis("off")
        axes[row, 3].axis("off")
        axes[row, 3].text(.02, .72, config.name.replace("_", " "), fontsize=13,
                          fontweight="bold", va="top")
        axes[row, 3].text(.02, .5, config.educational_message, fontsize=11.5,
                          va="top", wrap=True)
    figure.tight_layout(h_pad=1, w_pad=.6)
    path = output_dir / "ACT1_REFINED_CONTACT_SHEET.png"
    metadata_path = output_dir / "ACT1_REFINED_CONTACT_SHEET.json"
    figure.savefig(path, dpi=120, facecolor="white"); plt.close(figure)
    metadata_path.write_text(json.dumps({
        "schema_version": 1, "representative_frames": records,
        "artifact_sha256": _sha256(path), "rendering_launches_optimizer": False,
    }, indent=2), encoding="utf-8")
    return path, metadata_path


def export_refined_manifest(manifest: Path, output_dir: Path) -> list[tuple[Path, Path, Path]]:
    """Render refined scenes exclusively from reviewed or prepared records."""
    configs = load_refined_manifest(manifest)
    cache: dict[str, PlaybackData] = {}

    def load(path: str) -> PlaybackData:
        if path not in cache:
            cache[path] = load_playback_data(manifest.parent / path)
        return cache[path]

    style = LectureStyle()
    exports = []
    for config in configs:
        source_paths: dict[str, str] = {}
        if config.kind == "comparison":
            assert config.comparison_data_paths is not None
            datasets = {method: load(path) for method, path in config.comparison_data_paths.items()}
            source_paths = {f"comparison_{method}": str(manifest.parent / path)
                            for method, path in config.comparison_data_paths.items()}
            exports.append(_comparison_export(config, output_dir, datasets, source_paths))
            continue
        data = None
        if config.data_path is not None:
            data = load(config.data_path)
            source_paths["playback"] = str(manifest.parent / config.data_path)
        operations = None
        if config.operation_log_path is not None:
            operation_path = manifest.parent / config.operation_log_path
            operations = load_operation_log(operation_path)
            source_paths["operation_log"] = str(operation_path)
        if config.kind == "random_mechanics":
            movie, figure, draw, details = _random_movie(config, data, style)
        elif config.kind == "local_mechanics":
            movie, figure, draw, details = _local_movie(config, data, style)
        elif config.kind == "ga_mechanics":
            movie, figure, draw, details = _ga_movie(config, data, operations or [], style)
        elif config.kind == "distance_barrier":
            movie, figure, draw, details = _barrier_movie(config, style)
            source_paths["map_source"] = str(manifest.parent / "buglab/maps.py")
        elif config.kind == "rectangular_optimizer":
            movie, figure, draw, details = _rectangular_optimizer_movie(config, data, style)
        else:
            raise ValueError(f"Unknown refined scene kind {config.kind!r}")
        exports.append(_save_custom(
            config, output_dir, movie, figure, draw, details, source_paths
        ))
    _contact_sheet(configs, output_dir)
    return exports


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Export focused Act I refinement scenes")
    parser.add_argument("--manifest", type=Path, default=Path("act1_refined_scenes.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/act1_refined"))
    args = parser.parse_args(argv)
    for gif, poster, metadata in export_refined_manifest(args.manifest, args.output):
        print(f"gif: {gif.resolve()}")
        print(f"poster: {poster.resolve()}")
        print(f"metadata: {metadata.resolve()}")


if __name__ == "__main__":
    main()
