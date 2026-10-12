"""Deterministic static crossover slides for the main Act I lecture sequence."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

from matplotlib import patches, pyplot as plt
import numpy as np

from .environment import World, simulate
from .scenes import LectureStyle
from .visualization import MUTATION_COLOR, PARENT_A_COLOR, PARENT_B_COLOR, field


CUT_INDEX = 4
MUTATED_INDICES = (5,)
PARENT_A_DEGREES = np.asarray((6, 10, -2, -8, 4, 4, -6, -5), dtype=float)
PARENT_B_DEGREES = np.asarray((-6, -10, 4, 10, -2, -4, 6, 5), dtype=float)
CHILD_BEFORE_DEGREES = np.concatenate((PARENT_A_DEGREES[:CUT_INDEX],
                                       PARENT_B_DEGREES[CUT_INDEX:]))
FINAL_CHILD_DEGREES = CHILD_BEFORE_DEGREES.copy()
FINAL_CHILD_DEGREES[5] = -22


def crossover_world() -> World:
    """Return the obstacle-free illustrative world used by every static frame."""
    return World(width=88, height=64, start=(12, 32), goal=(76, 33), obstacles=(),
                 steps=8, step_length=8, max_turn=np.deg2rad(24),
                 map_name="illustrative_crossover")


def crossover_genomes_radians() -> dict[str, np.ndarray]:
    """Return fresh radian arrays so callers cannot mutate the teaching specification."""
    return {
        "parent_a": np.deg2rad(PARENT_A_DEGREES),
        "parent_b": np.deg2rad(PARENT_B_DEGREES),
        "child_before_mutation": np.deg2rad(CHILD_BEFORE_DEGREES),
        "final_child": np.deg2rad(FINAL_CHILD_DEGREES),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gene_row(ax, values: np.ndarray, y: float, label: str, colors: list[str]) -> None:
    ax.text(-.7, y, label, ha="right", va="center", fontsize=14, fontweight="bold")
    for index, (value, color) in enumerate(zip(values, colors)):
        ax.add_patch(patches.Rectangle((index - .42, y - .28), .84, .56,
                                       facecolor=color, edgecolor="white", lw=1.5))
        ax.text(index, y, f"{value:+.0f}°", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        ax.text(index, y - .43, str(index), ha="center", va="top", fontsize=9,
                color="#5f6872")


def _gene_axes(figure, rows: int = 3):
    bounds = (.48, .14, .48, .62) if rows == 3 else (.48, .18, .48, .64)
    ax = figure.add_axes(bounds)
    ax.set_xlim(-2.0, 7.65)
    ax.set_ylim(-.8, rows - .15)
    ax.axis("off")
    return ax


def _map_axes(figure, world: World):
    ax = figure.add_axes((.055, .17, .38, .66))
    field(ax, world, show_checkpoints=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax


def _plot_trace(ax, trace, *, color: str, label: str, width: float = 3,
                linestyle: str = "-") -> None:
    ax.plot(trace.positions[:, 0], trace.positions[:, 1], color=color, lw=width,
            ls=linestyle, label=label)


def _base_figure(style: LectureStyle, title: str, subtitle: str):
    figure = plt.figure(figsize=(style.canvas_width, style.canvas_height), facecolor="white")
    figure.suptitle(title, fontsize=21, fontweight="bold", y=.95)
    figure.text(.5, .895, subtitle, ha="center", fontsize=13, color="#4d5964")
    return figure


def _render_frame(frame_id: str, style: LectureStyle):
    world = crossover_world()
    genomes = crossover_genomes_radians()
    traces = {name: simulate(world, genome) for name, genome in genomes.items()}

    if frame_id == "04a_crossover_parents":
        figure = _base_figure(style, "Кроссовер: две родительские программы",
                              "Каждый ген — относительный поворот; траектории получены симулятором")
        map_ax = _map_axes(figure, world)
        _plot_trace(map_ax, traces["parent_a"], color=PARENT_A_COLOR, label="Родитель A")
        _plot_trace(map_ax, traces["parent_b"], color=PARENT_B_COLOR, label="Родитель B")
        map_ax.legend(loc="lower left", fontsize=11)
        genes_ax = _gene_axes(figure, 2)
        _gene_row(genes_ax, PARENT_A_DEGREES, 1.25, "Родитель A",
                  [PARENT_A_COLOR] * 8)
        _gene_row(genes_ax, PARENT_B_DEGREES, .15, "Родитель B",
                  [PARENT_B_COLOR] * 8)
        figure.text(.72, .12, "Индексы генов показаны под ячейками", ha="center", fontsize=11)
        message = "Две короткие последовательности команд задают две разные траектории."
    elif frame_id == "04b_crossover_child":
        figure = _base_figure(style, "Одноточечный кроссовер",
                              "Срез перед геном 4: начало от A, окончание от B")
        map_ax = _map_axes(figure, world)
        _plot_trace(map_ax, traces["child_before_mutation"], color="#4f6577",
                    label="Ребёнок до мутации")
        map_ax.legend(loc="lower left", fontsize=11)
        genes_ax = _gene_axes(figure, 3)
        _gene_row(genes_ax, PARENT_A_DEGREES, 2.0, "Родитель A",
                  [PARENT_A_COLOR] * 8)
        _gene_row(genes_ax, PARENT_B_DEGREES, 1.05, "Родитель B",
                  [PARENT_B_COLOR] * 8)
        inherited = [PARENT_A_COLOR] * CUT_INDEX + [PARENT_B_COLOR] * (8 - CUT_INDEX)
        _gene_row(genes_ax, CHILD_BEFORE_DEGREES, .05, "Ребёнок", inherited)
        genes_ax.vlines(CUT_INDEX - .5, -.32, 2.30, color="#20262d", lw=2, ls="--")
        genes_ax.text(CUT_INDEX - .5, 2.38, "точка среза", ha="center", fontsize=11)
        figure.text(.72, .105, "Кроссовер соединяет команды, а не куски геометрических путей.",
                    ha="center", fontsize=11.5, fontweight="bold")
        message = "Ребёнок наследует команды по обе стороны одной точки среза."
    elif frame_id == "04c_crossover_mutation":
        figure = _base_figure(style, "Мутация: небольшое изменение программы",
                              "Меняется один явно отмеченный ген; остальные команды сохранены")
        map_ax = _map_axes(figure, world)
        _plot_trace(map_ax, traces["child_before_mutation"], color="#8a96a3",
                    label="До мутации", width=2, linestyle="--")
        _plot_trace(map_ax, traces["final_child"], color=MUTATION_COLOR,
                    label="После мутации")
        map_ax.legend(loc="lower left", fontsize=11)
        genes_ax = _gene_axes(figure, 2)
        inherited = [PARENT_A_COLOR] * CUT_INDEX + [PARENT_B_COLOR] * (8 - CUT_INDEX)
        _gene_row(genes_ax, CHILD_BEFORE_DEGREES, 1.25, "До мутации", inherited)
        final_colors = inherited.copy()
        final_colors[5] = MUTATION_COLOR
        _gene_row(genes_ax, FINAL_CHILD_DEGREES, .15, "После", final_colors)
        genes_ax.annotate("ген 5: −4° → −22°", xy=(5, .46), xytext=(5, .82), ha="center",
                          fontsize=11, color=MUTATION_COLOR, fontweight="bold",
                          arrowprops={"arrowstyle": "->", "color": MUTATION_COLOR})
        message = "Мутация меняет одну команду и тем самым изменяет последующую траекторию."
    elif frame_id == "04d_crossover_outcome":
        figure = _base_figure(style, "Новая программа — новая траектория",
                              "Кроссовер и мутация создают вариант для последующей оценки")
        map_ax = figure.add_axes((.08, .17, .56, .66))
        field(map_ax, world, show_checkpoints=False)
        _plot_trace(map_ax, traces["parent_a"], color=PARENT_A_COLOR, label="Родитель A", width=2)
        _plot_trace(map_ax, traces["parent_b"], color=PARENT_B_COLOR, label="Родитель B", width=2)
        _plot_trace(map_ax, traces["child_before_mutation"], color="#8a96a3",
                    label="Ребёнок до мутации", width=2, linestyle="--")
        _plot_trace(map_ax, traces["final_child"], color=MUTATION_COLOR,
                    label="Итоговый ребёнок", width=3.5)
        map_ax.legend(loc="lower left", fontsize=10)
        text_ax = figure.add_axes((.69, .23, .27, .52)); text_ax.axis("off")
        text_ax.text(0, .88, "1. Срез", fontsize=16, fontweight="bold", color="#20262d")
        text_ax.text(0, .76, "Команды двух родителей\nобразуют нового ребёнка.", fontsize=12)
        text_ax.text(0, .51, "2. Мутация", fontsize=16, fontweight="bold", color=MUTATION_COLOR)
        text_ax.text(0, .39, "Одна команда получает\nслучайное изменение.", fontsize=12)
        text_ax.text(0, .14, "Результат ещё не обязан\nбыть лучше — его надо оценить.",
                     fontsize=12, fontweight="bold")
        message = "Генетические операции создают кандидата, но не гарантируют улучшение."
    else:
        raise ValueError(f"Unknown crossover frame {frame_id!r}")
    figure.text(.5, .035, message, ha="center", fontsize=12.5, fontweight="bold")
    return figure, world, traces


def export_crossover_slides(output_dir: Path,
                            style: LectureStyle = LectureStyle()) -> list[tuple[Path, Path, Path]]:
    """Export four 16:9 PNG/SVG teaching frames and reproducibility sidecars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = (
        "04a_crossover_parents",
        "04b_crossover_child",
        "04c_crossover_mutation",
        "04d_crossover_outcome",
    )
    exports = []
    for frame_id in frame_ids:
        figure, world, traces = _render_frame(frame_id, style)
        png_path = output_dir / f"{frame_id}.png"
        svg_path = output_dir / f"{frame_id}.svg"
        json_path = output_dir / f"{frame_id}.json"
        figure.savefig(png_path, dpi=150, facecolor="white")
        figure.savefig(svg_path, facecolor="white")
        plt.close(figure)
        payload = {
            "schema_version": 1,
            "scene_identifier": frame_id,
            "source_type": "synthetic illustrative crossover with simulator-derived trajectories",
            "illustrative": True,
            "world": {
                "width": world.width, "height": world.height, "start": world.start,
                "goal": world.goal, "steps": world.steps, "step_length": world.step_length,
                "max_turn_radians": world.max_turn,
            },
            "operation": {
                "cut_index": CUT_INDEX, "mutated_indices": MUTATED_INDICES,
                "parent_a_degrees": PARENT_A_DEGREES.tolist(),
                "parent_b_degrees": PARENT_B_DEGREES.tolist(),
                "child_before_mutation_degrees": CHILD_BEFORE_DEGREES.tolist(),
                "final_child_degrees": FINAL_CHILD_DEGREES.tolist(),
                "genomes_radians": {name: genome.tolist()
                                    for name, genome in crossover_genomes_radians().items()},
            },
            "simulation": {
                name: {"reached_goal": trace.reached, "collisions": trace.collisions,
                       "trajectory_points": len(trace.positions)}
                for name, trace in traces.items()
            },
            "render": {"canvas_pixels": [1920, 1080], "png_sha256": _digest(png_path),
                       "svg_sha256": _digest(svg_path)},
            "rendering_launches_optimizer": False,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        exports.append((png_path, svg_path, json_path))
    return exports


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Export deterministic Act I crossover slide frames")
    parser.add_argument("--output", type=Path, default=Path("outputs/act1_final"))
    args = parser.parse_args(argv)
    for png, svg, metadata in export_crossover_slides(args.output):
        print(f"png: {png.resolve()}")
        print(f"svg: {svg.resolve()}")
        print(f"metadata: {metadata.resolve()}")


if __name__ == "__main__":
    main()
