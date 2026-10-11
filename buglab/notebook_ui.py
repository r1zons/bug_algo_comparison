"""Optional ipywidgets interface over the importable playback layer."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from .fitness import FITNESS_NAMES
from .maps import MAPS
from .optimizers import METHODS
from .playback import (
    ExperimentCache,
    PlaybackConfig,
    PlaybackData,
    export_generation_png,
    export_summary_png,
    progress_figure,
    trajectory_figure,
)


@dataclass
class NotebookFallback:
    """Helpful static-mode result returned when widgets are unavailable."""

    message: str
    cache: ExperimentCache

    def _repr_markdown_(self) -> str:
        return f"**Interactive controls unavailable.** {self.message}"

    def render_static(self, data: PlaybackData, snapshot_index: int = 0):
        """Return the same static figures used by the widget interface."""
        return trajectory_figure(data, snapshot_index), progress_figure(data, snapshot_index)


class PlaybackWidget:
    """Stateful notebook controller; only the Run button starts optimization."""

    def __init__(
        self,
        cache: ExperimentCache | None = None,
        *,
        output_dir: Path = Path("outputs_notebook"),
    ) -> None:
        import ipywidgets as widgets

        self.widgets = widgets
        self.cache = cache or ExperimentCache()
        self.output_dir = output_dir
        self.current: PlaybackData | None = None
        self._syncing = False

        self.map = widgets.Dropdown(options=tuple(MAPS), value="default", description="Map")
        self.method = widgets.Dropdown(options=tuple(METHODS), value="ga", description="Method")
        self.fitness = widgets.Dropdown(options=FITNESS_NAMES, value="geodesic", description="Fitness")
        self.seed = widgets.IntText(value=0, description="Seed")
        self.generations = widgets.BoundedIntText(value=8, min=1, max=200, description="Batches")
        self.population = widgets.BoundedIntText(value=16, min=1, max=256, description="Population")
        self.run_button = widgets.Button(description="Run / load", button_style="primary", icon="play")
        self.cached_run = widgets.Dropdown(options=[], description="Cached run")

        self.play = widgets.Play(value=0, min=0, max=0, step=1, interval=700, description="Play")
        self.generation = widgets.IntSlider(value=0, min=0, max=0, step=1,
                                            description="Generation", continuous_update=False)
        widgets.jslink((self.play, "value"), (self.generation, "value"))
        self.candidate = widgets.IntSlider(value=0, min=0, max=0, step=1,
                                           description="Individual", continuous_update=False)

        self.show_all = widgets.Checkbox(value=True, description="All trajectories")
        self.show_selected = widgets.Checkbox(value=True, description="Selected individual")
        self.show_best = widgets.Checkbox(value=True, description="Best-so-far")
        self.show_collisions = widgets.Checkbox(value=False, description="Collision markers")
        self.highlight_success = widgets.Checkbox(value=True, description="Highlight successes")
        self.show_checkpoints = widgets.Checkbox(
            value=False, description="Show checkpoint markers (display only)"
        )
        self.top_k = widgets.IntSlider(value=0, min=0, max=16, step=1,
                                       description="Top-k (0=all)", continuous_update=False)

        self.export_generation_button = widgets.Button(description="Export generation PNG", icon="image")
        self.export_summary_button = widgets.Button(description="Export summary PNG", icon="save")
        self.status = widgets.HTML(value="Choose settings and press <b>Run / load</b>.")
        self.summary = widgets.HTML()
        self.inspector = widgets.HTML()
        self.output = widgets.Output()

        settings = widgets.VBox([
            widgets.HTML("<h3>Experiment settings</h3>"),
            widgets.HBox([self.map, self.method, self.fitness]),
            widgets.HBox([self.seed, self.generations, self.population, self.run_button]),
            self.cached_run,
        ])
        playback = widgets.VBox([
            widgets.HTML("<h3>Recorded playback</h3>"),
            widgets.HBox([self.play, self.generation, self.candidate]),
            widgets.HBox([self.show_all, self.show_selected, self.show_best,
                          self.show_collisions]),
            widgets.HBox([self.highlight_success, self.show_checkpoints, self.top_k]),
            widgets.HBox([self.export_generation_button, self.export_summary_button]),
        ])
        panels = widgets.HBox([self.summary, self.inspector])
        self.widget = widgets.VBox([settings, playback, self.status, panels, self.output])

        self.run_button.on_click(self._run_clicked)
        self.export_generation_button.on_click(self._export_generation)
        self.export_summary_button.on_click(self._export_summary)
        self.cached_run.observe(self._cached_selected, names="value")
        for control in (self.generation, self.candidate, self.show_all, self.show_selected,
                        self.show_best, self.show_collisions, self.highlight_success,
                        self.show_checkpoints, self.top_k):
            control.observe(self._playback_changed, names="value")
        for control in (self.map, self.method, self.fitness, self.seed,
                        self.generations, self.population):
            control.observe(self._settings_changed, names="value")
        self._refresh_cached_options()
        if self.cache.values():
            self._set_current(self.cache.values()[0])

    def display(self) -> None:
        from IPython.display import display
        display(self.widget)

    def _config(self) -> PlaybackConfig:
        return PlaybackConfig(
            map_name=self.map.value,
            method=self.method.value,
            fitness_name=self.fitness.value,
            seed=self.seed.value,
            generations=self.generations.value,
            population=self.population.value,
        )

    def _refresh_cached_options(self) -> None:
        options = [(data.config.label, data.config) for data in self.cache.values()]
        self._syncing = True
        self.cached_run.options = options
        self._syncing = False

    def _settings_changed(self, change: dict[str, Any]) -> None:
        if not self._syncing:
            self.status.value = (
                "<span style='color:#8a5b19'>Settings changed. Press <b>Run / load</b> "
                "to create or select correctly labeled data.</span>"
            )

    def _run_clicked(self, button: Any) -> None:
        config = self._config()
        cached = self.cache.get(config)
        self.status.value = "Running experiment…" if cached is None else "Loading cached experiment…"
        data = self.cache.run(config)
        self._refresh_cached_options()
        self._set_current(data)
        action = "Loaded cached" if cached is not None else "Completed and cached"
        self.status.value = f"<b>{action}:</b> {escape(config.label)}"

    def _cached_selected(self, change: dict[str, Any]) -> None:
        if self._syncing or change["new"] is None:
            return
        data = self.cache.get(change["new"])
        if data is not None:
            self._set_current(data)
            self.status.value = f"<b>Selected cached run:</b> {escape(data.config.label)}"

    def _set_current(self, data: PlaybackData) -> None:
        self.current = data
        self._syncing = True
        self.map.value = data.config.map_name
        self.method.value = data.config.method
        self.fitness.value = data.config.fitness_name
        self.seed.value = data.config.seed
        self.generations.value = data.config.generations
        self.population.value = data.config.population
        self.generation.description = data.iteration_kind.title()
        self.generation.min = data.generations[0]
        self.generation.max = data.generations[-1]
        self.generation.value = data.generations[0]
        self.play.min = data.generations[0]
        self.play.max = data.generations[-1]
        self.play.value = data.generations[0]
        self.candidate.max = data.config.population - 1
        self.candidate.value = data.selected_candidate(0)
        self.top_k.max = data.config.population
        self.cached_run.value = data.config
        self._syncing = False
        self._render()

    def _snapshot_index(self) -> int:
        assert self.current is not None
        return self.current.snapshot_index(self.generation.value)

    def _playback_changed(self, change: dict[str, Any]) -> None:
        if self._syncing or self.current is None:
            return
        if change["owner"] is self.generation:
            index = self._snapshot_index()
            self._syncing = True
            self.candidate.value = self.current.selected_candidate(index)
            self._syncing = False
        self._render()

    def _render(self) -> None:
        if self.current is None:
            return
        from IPython.display import display
        index = self._snapshot_index()
        candidate = self.candidate.value
        figure = trajectory_figure(
            self.current,
            index,
            selected_candidate=candidate,
            show_all=self.show_all.value,
            show_selected=self.show_selected.value,
            show_best_so_far=self.show_best.value,
            show_collisions=self.show_collisions.value,
            highlight_success=self.highlight_success.value,
            show_checkpoints=self.show_checkpoints.value,
            top_k=self.top_k.value or None,
        )
        progress = progress_figure(self.current, index)
        with self.output:
            self.output.clear_output(wait=True)
            display(figure)
            display(progress)
        plt.close(figure);plt.close(progress)
        summary = self.current.summary(index, candidate)
        individual = self.current.individual_summary(index, candidate)
        self.summary.value = self._summary_html(summary, self.current.method_description)
        self.inspector.value = self._inspector_html(individual)

    @staticmethod
    def _summary_html(summary: dict[str, Any], method_description: str) -> str:
        fields = (
            "method", "map", "fitness", "seed", "iteration_kind", "iteration",
            "selected_candidate_index", "selected_candidate_score",
            "best_by_fitness_candidate_index", "best_by_fitness_score",
            "best_so_far_score", "best_so_far_first_evaluation",
            "fastest_successful_evaluation", "fastest_successful_steps_to_goal",
            "selected_goal_reached", "steps_to_goal", "collisions", "path_length",
            "evaluations", "environment_steps",
        )
        rows = "".join(
            f"<tr><th style='text-align:left;padding-right:12px'>{escape(key.replace('_',' '))}</th>"
            f"<td>{escape(str(summary[key]))}</td></tr>" for key in fields
        )
        return (
            "<div style='padding:8px'><h4>Run summary</h4>"
            f"<p style='max-width:480px'>{escape(method_description)}</p>"
            f"<table>{rows}</table></div>"
        )

    @staticmethod
    def _inspector_html(individual: dict[str, Any]) -> str:
        return (
            "<div style='padding:8px;max-width:560px'><h4>Individual inspector</h4>"
            f"<b>candidate:</b> {individual['candidate']} &nbsp; "
            f"<b>score:</b> {individual['score']:.3f} &nbsp; "
            f"<b>genome length:</b> {individual['genome_length']}<br>"
            f"<b>outcome:</b> goal={individual['goal_reached']}, "
            f"collisions={individual['collisions']}, final distance={individual['final_distance']:.2f}<br>"
            f"<b>angle preview:</b> <code>{escape(individual['angles_preview'])}</code></div>"
        )

    def _export_generation(self, button: Any) -> None:
        if self.current is None:
            return
        index = self._snapshot_index()
        path = self.output_dir / (
            f"{self.current.config.method}_{self.current.iteration_kind}_{self.generation.value}.png"
        )
        export_generation_png(
            self.current,index,path,selected_candidate=self.candidate.value,
            show_all=self.show_all.value,show_selected=self.show_selected.value,
            show_best_so_far=self.show_best.value,show_collisions=self.show_collisions.value,
            highlight_success=self.highlight_success.value,
            show_checkpoints=self.show_checkpoints.value,top_k=self.top_k.value or None,
        )
        self.status.value = f"Saved generation view to <code>{escape(str(path))}</code>"

    def _export_summary(self, button: Any) -> None:
        if self.current is None:
            return
        index = self._snapshot_index()
        path = self.output_dir / f"{self.current.config.method}_run_summary.png"
        export_summary_png(self.current,index,path,self.candidate.value)
        self.status.value = f"Saved run summary to <code>{escape(str(path))}</code>"


def create_playback_widget(
    cache: ExperimentCache | None = None,
    *,
    output_dir: Path = Path("outputs_notebook"),
) -> PlaybackWidget | NotebookFallback:
    """Create the widget UI, or a documented static fallback without ipywidgets."""
    try:
        import ipywidgets  # noqa: F401
        import IPython  # noqa: F401
    except ImportError:
        return NotebookFallback(
            "Install optional notebook support with `pip install -r requirements-notebook.txt`. "
            "Static playback figures and PNG exports remain available from `buglab.playback`.",
            cache or ExperimentCache(),
        )
    return PlaybackWidget(cache, output_dir=output_dir)
