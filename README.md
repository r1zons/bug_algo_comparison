# Лаборатория эволюции жуков

A reproducible teaching simulator with six maps, explicit reward strategies, and optimizers sharing the same beetle physics.

## Быстрый запуск

```bash
pip install -r requirements.txt
python -m buglab --generations 90 --population 64 --seeds 0 1 2 --gif
python -m buglab --map spiral --fitness checkpoints --generations 30 --population 32 --seeds 0
python -m buglab --map spiral --gallery --compare-fitness euclidean geodesic checkpoints
python -m buglab --generations 8 --population 16 --seeds 0 --methods ga --operator-story
```

Файлы в `outputs/`: `ga_snapshots.png` (поколения), `ga_evolution.gif`, `comparison.png`, `history.csv`, `summary.csv`, лучшие геномы `.npy`.

## Физика (фиксирована для всех методов)

- Каждая особь: геном `g = (Δθ_1, …, Δθ_T)` длины `T`.
- На каждом шаге жук поворачивает на `Δθ_t` (ограничение `max_turn`) и движется ровно на `step_length` в текущем направлении.
- Если путь пересекает препятствие или границу, перемещение блокируется, но поворот выполняется.
- `T=90` шагов — один эпизод; GA начинает новое поколение только после завершения оценки всех особей.
- Функция `fitness`: прогресс по препятствие-осведомлённому расстоянию до цели, бонус достижения, небольшой штраф столкновений и времени достижения цели. Кратчайший путь используется **только для оценки**, оптимизатор получает лишь скалярное значение; координаты маршрута не передаются оптимизатору.
- Карта расстояний из 8-соседнего Dijkstra по сетке — это дизайнерское решение fitness, а не решение задачи GA; оно даёт дополнительную информацию о препятствиях и должно учитываться при интерпретации эксперимента.

## Архитектура

```text
Map -> World + simulate(genome) -> EpisodeTrace -> FitnessPolicy -> scalar score
                                  |                               |
                                  +-> common outcome metrics      v
Optimizer.ask() -> genome batch --------------------------> Optimizer.tell()

Experiment runner -> history/snapshots -> visualization (PNG/GIF/CSV)
```

### Модули

- `geometry.py` and `maps.py`: reusable collision geometry and map definitions.
- `environment.py`: deterministic physics and score-independent episode traces.
- `fitness.py`: pluggable scalar scoring policies; the historical score remains the default.
- `metrics.py`: reward-independent physical outcome metrics.
- `optimizers.py`: единый интерфейс `ask()` / `tell()`, GA, RandomSearch, LocalSearch.
- `experiment.py`: воспроизводимые сравнения, бюджет, результаты.
- `visualization.py`: независимый рендер кадров, графиков и GIF.
- `playback.py`: compact cached runs, lazy deterministic trajectory replay, and PNG exports.
- `notebook_ui.py`: optional `ipywidgets` controls over recorded playback data.
- `__main__.py`: CLI.

**Чтобы добавить метод:** реализуйте `ask()` (возврат массива `population × T`) и `tell(genomes, scores)`; зарегистрируйте класс в `METHODS`. Никаких изменений в физике или рендере не требуется.

## Что показывать в докладе

1. Поле/цель/препятствие.
2. Геном из чисел (углы).
3. Разные жуки, разные траектории первого поколения.
4. One-point crossover and mutation from the deterministic synthetic example and bounded real GA operation log.
5. Реальные поколения и лучший маршрут.
6. График GA/random/local при одинаковом числе вызовов fitness.
7. Обсудить роль функции fitness, ограничений, операторов и бюджета.

## Научная оговорка

График по 3 seeds — **демонстрационный, не доказательство превосходства**. Для надёжного сравнения увеличьте число seeds до 20–30, контролируйте параметры и сравнивайте дополнительно успех достижения цели. Методы имеют разную вычислительную цену помимо fitness и разные гиперпараметры, поэтому сравнение по одному количеству evaluations не исчерпывает сопоставимость.

Use `--map` to select `default`, `open_field`, `two_bridges`, `spiral`, `narrow_gate`, or `dead_end`. Use `--fitness` to select `sparse_goal`, `euclidean`, `final_euclidean`, `success_distance`, `collision_penalty`, `checkpoints`, `geodesic`, or `smoothness`.

### Reward information levels

- `sparse_goal` returns one only after physical goal arrival and zero otherwise. Its distance field is diagnostic only.
- `euclidean` is the historical closest-approach score: negative minimum straight-line goal distance attained anywhere along the episode. It knows the goal coordinate but has no obstacle-aware route information. This is the recommended default for introductory lecture scenes.
- `final_euclidean` scores only the final simulated position. It is separate so reports cannot silently confuse terminal distance with the legacy closest-approach score.
- `geodesic` uses a precomputed obstacle-aware shortest-distance field and therefore injects additional map-specific structure.
- `checkpoints` uses an ordered designer-authored corridor and injects even more explicit map-specific guidance.

Physical success is always defined by simulator goal arrival, independently of the reward policy. The historical CLI and benchmark defaults remain `geodesic` for reproducibility; existing results are not overwritten.

`--compare-fitness` runs the same optimizer, map, seeds, and evaluation budget for each requested strategy and writes `fitness_comparison.csv` and `fitness_comparison.png`. Those files compare only physical outcomes: success rate, steps to goal, collisions, traveled distance, final distance, evaluation count, and environment steps. Raw fitness values are intentionally excluded because each reward defines a different scale.

Geodesic distance supplies an obstacle-aware distance grid, while checkpoints supply an ordered designer-authored corridor. Both provide additional domain knowledge unavailable to Euclidean proximity. On `spiral`, moving visually toward the goal can end at a wall; valid progress must wind through the corridor and may initially look worse in straight-line distance.

## Crossover and mutation story

Use `--operator-story` with a GA run to produce:

- `synthetic_crossover.png`: an eight-gene deterministic example in degrees;
- `ga_operation_log.json`: bounded replayable events stored in radians;
- `real_operation_improvement.png`;
- `real_operation_neutral.png`;
- `real_operation_degradation.png`.

The synthetic example uses a fixed cut at index 4 and fixed mutations at indices 2 and 6. Blue and orange genes identify parent inheritance; magenta identifies mutation. Crossover combines open-loop control instructions, not geometrically connected path segments.

Real events are aligned across consecutive `ask()`/`tell()` batches. Tracing is disabled by default and bounded by `--operation-log-limit` (default 256). Enabling it does not add evaluations, replay trajectories during optimization, or consume random values. A requested category that is absent gets an honestly labeled placeholder image.

Classification uses `max(0.25, 1% × |better parent fitness|)` as the near-neutral tolerance. Improvement must beat the better parent; degradation must be worse than both parents. Events between the parents are classified as mixed and are not presented as degradation.

## Interactive notebook playback

Install the optional notebook dependencies and open the thin demo notebook:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab buglab_demo.ipynb
```

The notebook loads `examples/task4_demo.npz`, a small deterministic GA run, and can explicitly run or cache other map, method, fitness, and seed combinations. The **Run / load** button is the only control that starts optimization. Playing generations, selecting an individual, changing overlays, and exporting figures only replay stored genomes through deterministic physics; they do not consume optimizer evaluations or rebuild the map distance field.

Use **Export generation PNG** for the selected trajectory view and **Export summary PNG** for a compact run report. Exports are written to `outputs_notebook/` by default. The same interface labels GA populations, random-search batches, and local-search incumbents/neighbors consistently. Raw fitness progress is meaningful only within the selected fixed fitness policy; use the shared outcome panels when comparing different reward functions.

If `ipywidgets` is unavailable, `create_playback_widget()` returns a documented static fallback. The importable functions in `buglab.playback` can still load the example cache and render or export figures with Matplotlib.

## Reproducible benchmarks and lecture exports

The fast preset is intentionally small and is suitable for CI or checking the complete export pipeline:

```bash
python -m buglab.benchmark --preset smoke --output outputs/task5_smoke
python -m buglab.benchmark --preset smoke --media --case-study --output outputs/task5_lecture
```

The second command additionally creates one movement GIF per optimizer, a synchronized comparison GIF, a side-by-side PNG, and the spiral/two-bridges fitness-design study. Rendering consumes recorded snapshots and never reruns optimization.

The full preset uses 25 seeds and 4,160 candidate evaluations per method and is not run by default:

```bash
python -m buglab.benchmark --preset full --output outputs/task5_full
```

All settings can be overridden with `--map`, `--fitness`, `--methods`, `--seeds`, `--population`, and `--evaluations`. The evaluation budget must be divisible by the population size. See [BENCHMARKS.md](BENCHMARKS.md) for metric definitions and interpretation limits.

Task 5 changes `first_success_evaluations` from the historical end-of-batch approximation to the exact one-based candidate ordinal. `first_success_batch_end_evaluations` preserves the old value. GA iterations are called generations; random-search and local-search iterations are called batches.

## Pilot lecture scenes

The declarative pilot manifest uses a recorded Euclidean-fitness cache and launches no optimizer while rendering:

```bash
python -m buglab.scenes \
  --manifest pilot_scenes.json \
  --data examples/pilot_lecture_euclidean.npz \
  --output outputs/pilot_scenes
```

Scene speed, duration, deterministic candidate selection, output format, and checkpoint-marker visibility are controlled in `pilot_scenes.json`. GIF is always supported; MP4 is accepted when Matplotlib can find `ffmpeg`. Each animation has a JSON sidecar with scene configuration and source provenance. See [ANIMATION_PLAN.md](ANIMATION_PLAN.md) for the complete Act I storyboard.

The notebook checkbox is labeled **Show checkpoint markers (display only)** and defaults off. It changes only the renderer; selecting checkpoint fitness remains a separate explicit experiment setting.

## Final Act I lecture package

The polished Act I sequence contains eight focused 16:9 GIFs and four static crossover/mutation frames. The original synthetic Scene 04 GIF remains optional comparison material. Prepare source experiments explicitly, then render only from saved data:

```bash
python -m buglab.act1 --output examples/act1
python -m buglab.act1_bundle \
  --manifest act1_final_scenes.json \
  --output outputs/act1_final
```

See [ACT1_ANIMATION_INDEX.md](ACT1_ANIMATION_INDEX.md) for the lecture order, source of every scene, reproduction details, and educational limitations. The source manifest is `act1_final_scenes.json`; each GIF and static frame has a JSON provenance sidecar, and GIF sidecars include the exported artifact hash. Historical outputs remain under `outputs/act1_before_polish/`.
