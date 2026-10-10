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

Use `--map` to select `default`, `open_field`, `two_bridges`, `spiral`, `narrow_gate`, or `dead_end`. Use `--fitness` to select `euclidean`, `success_distance`, `collision_penalty`, `checkpoints`, `geodesic`, or `smoothness`.

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
