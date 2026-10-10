# Architecture: maps, physics, fitness, and optimizers

## Separation of responsibilities

```
 MapDefinition ──> World + simulate(genome) ──> EpisodeTrace
                         │                         │
                         │                         ├──> outcome_metrics
                         │                         │    (reward-independent)
                         │                         └──> FitnessPolicy ──> scalar
                         │                                               │
                         └───────────────────────────────────────────────┤
                                                                         │ tell
                    ┌────────────────────────────────────────────────────┘
                    │ Experiment runner (equal evaluation budget, seeds)
                    └───────────────┬─────────────────────
                                    │ ask
                   ┌────────────────┼─────────────────┐
                   ▼                ▼                 ▼
                  GA          Random Search       Local Search
                   └────────────────┼─────────────────┘
                                    ▼
                    Recorded snapshots and outcomes → CSV / GIF / PNG
                                      │
                                      └──> cached playback → notebook widgets / PNG
```

## Interface contracts

- `MapDefinition`: immutable dimensions, start, goal, obstacles, and optional ordered checkpoints.
- `World`: one map combined with common motion/collision settings.
- `simulate(world, genome) -> EpisodeTrace`: deterministic physics with no reward calculation.
- `FitnessPolicy.score(world, trace) -> FitnessResult`: a scalar training signal.
- `evaluate(world, genome, fitness) -> Rollout`: compatibility composition of simulation, scoring, and common metrics.
- `Optimizer.ask() -> numpy.ndarray[population, T]` and `tell(genomes, scores)`: optimizers receive arrays and **scalar scores only**.
- `run_method(...) -> Run`: equal-budget orchestration, recorded configurations, snapshots, and outcomes.
- Visualization consumes recorded results and never changes optimization or physics.
- Playback stores genomes, scalar scores, history, and metadata; trajectories are reconstructed lazily with deterministic physics.

## Добавление алгоритма

1. Создать класс в `optimizers.py` с `ask()/tell()`.
2. Добавить его в словарь `METHODS`.
3. Выполнить `python -m buglab --methods ga random local new_method`.
4. Убедиться, что новый алгоритм получает одинаковый тип генома, одну функцию качества и сопоставимый бюджет.

Идеи для следующего шага: simulated annealing, CMA-ES (на вещественных геномах), evolution strategies без crossover, Bayesian optimization с компактным параметрическим геномом, novelty search.

## Если подключать NN / RL позже

Сравнение в пределах текущего набора предполагает **открытое управление**: заранее заданы все относительные повороты. Для нейросети лучше выделить контракт `Policy(observation) -> delta_angle`. Тогда цикл физики останется общим, а оптимизаторы смогут искать либо фиксированную последовательность действий, либо параметры адаптивной политики. Для честного сравнения с RL нужно отслеживать не только вызовы `evaluate`, но и **количество шагов среды**, а также отдельно тестировать на новых картах: open-loop траектория запоминает конкретную карту и не обобщается сама собой.

Это следующий этап архитектуры, **не реализованный в MVP**. Нельзя выдавать нынешнее сравнение GA и локального поиска за сравнение с RL.

## Task 1 extension boundaries

The original public API remains available, while the core now has explicit seams:

- `geometry.py` owns reusable collision primitives.
- `maps.py` owns immutable map geometry; `World.from_map()` combines it with physics settings.
- `environment.simulate()` produces a score-independent `EpisodeTrace`.
- `fitness.py` turns an episode trace into a scalar training signal. `evaluate()` remains the compatibility composition of simulation and scoring.
- `run_method()` accepts an optional fitness policy, but optimizers still receive only genome arrays and scalar scores.
- Stored snapshots contain bounded candidate genome/score batches, allowing later playback to replay physics lazily instead of retaining every trajectory from every generation.

The historical `World(fitness_mode=...)`, `evaluate(world, genome)`, CLI arguments, and default map remain compatible. CSV outputs add configuration and common outcome columns. `Rollout.min_geodesic` is retained as a compatibility field name even when another policy supplies its closest-distance diagnostic.

## Task 2 maps, rewards, and outcome comparisons

Named presets are `default`, `open_field`, `two_bridges`, `spiral`, `narrow_gate`, and `dead_end`. Each preset has a deterministic reference corridor that is reachable under the default 90-step horizon, 1.55 step length, 0.42-radian maximum relative turn, beetle radius, and segment collision rules. These references validate the task; optimizers do not receive them.

Named fitness strategies are `euclidean`, `success_distance`, `collision_penalty`, `checkpoints`, `geodesic`, and `smoothness`. Collision and smoothness policies compose configurable penalties with a base reward. Physical goal detection stays in `simulate()` and is identical for every reward.

Geodesic shaping uses an obstacle-aware grid and checkpoint shaping encodes an ordered useful corridor. Both inject map-specific domain knowledge. In particular, the spiral goal is visually close across its inner wall: Euclidean proximity can reward approaching that wall even though the valid route initially winds away from the goal. This is why raw scalar fitness values from different policies must never be compared.

Cross-policy reports use only goal reached, steps to goal, collision count, actual traveled path length, final Euclidean distance, fitness-evaluation count, and environment-step count. Success rate is the fraction of evaluated genomes that physically reached the goal. Representative route metrics use the fastest successful rollout when one exists; otherwise they describe the fitness-selected best failure.

## Task 3 genetic-operation provenance

`GeneticAlgorithm` supports an optional bounded operation trace. The first population has no genetic-operation events because it is sampled, not crossed. After a population is scored, the GA records the existing tournament selections, elite copies, crossover cut, and applied mutation deltas while constructing the next batch. It consumes no additional random values. On the next `tell()`, candidate indices attach the observed scalar scores to those pending events. The experiment runner independently attaches physical outcome metrics; map geometry and trajectories never enter the optimizer.

Every event explicitly distinguishes `elite` from `offspring`. Elite events have one source parent, no crossover cut, and no mutation. Offspring contain both source candidate IDs and scores, parent genomes, the pre-mutation crossover child, actual post-clipping mutation deltas, and the evaluated final child. Only selected examples replay trajectories during rendering, so the bounded log does not retain trajectory arrays.

Real teaching examples use a documented tolerance `max(0.25, 1% × |better parent fitness|)`:

- improvement: child exceeds the better parent by more than the tolerance;
- near-neutral: child is within the tolerance of the better parent;
- degradation: child is below the worse parent by more than the tolerance;
- mixed: child lies between its parents outside the neutral band.

The first recorded event in each requested category is selected deterministically. Missing categories remain explicitly absent and are never synthesized. Raw fitness comparisons are valid within one event because both parents and the child use the same map, physics, and fitness policy.

## Task 4 interactive playback

`PlaybackData` is the presentation-facing boundary around a completed `Run`. Each recorded generation contains candidate genomes and their already-observed scores, the best-so-far genome and score, evaluation count, run history, and immutable map/method/fitness metadata. It intentionally does not persist every trajectory. `candidate_rollout()` and `best_rollout()` replay a genome on first access and memoize that deterministic result for later scrubbing.

`ExperimentCache.run()` is the only notebook path that launches optimization. A complete `PlaybackConfig` is its cache key, so changing a map, optimizer, fitness policy, seed, or search size cannot relabel old results. Slider, play, candidate, overlay, and export changes operate only on the selected `PlaybackData`. The notebook records every batch for smooth playback; normal CLI snapshot behavior remains unchanged.

Compact `.npz` caches use numeric arrays plus JSON metadata and load with `allow_pickle=False`. The bundled `examples/task4_demo.npz` is an intentional small pedagogical artifact. Map geometry and the chosen fitness policy are reconstructed once when it is loaded; trajectory replay then reuses those objects.

`notebook_ui.py` is optional and imports `ipywidgets` only when the interface is created. Without notebook dependencies, the simulator, CLI, static Matplotlib playback, and PNG exports remain usable. The notebook itself contains initialization and teaching guidance only; playback and rendering logic stays in importable modules.

The progress display labels evaluation budget explicitly. Its raw-fitness panel applies to one fixed policy only. Cross-policy interpretation must use the adjacent shared outcomes, never numerical fitness values from different reward definitions.

## Интерпретация начального эксперимента

Предустановленная карта содержит два препятствия. Запуск `population=64, generations=65` даёт 4 160 проверок геномов на метод и seed. В трёх тестовых seeds успех обнаружен у GA в 3/3, у LocalSearch в 3/3, у RandomSearch в 1/3. В данном запуске LocalSearch часто находил успешную траекторию раньше GA. Это **демонстрация механики**, а не статистически значимый вывод.

Критично: `geodesic` fitness строится с помощью кратчайших расстояний на дискретной карте препятствий. Это даёт полезный информативный сигнал при обучении; в benchmark надо честно указать эту особенность и при необходимости сравнить с `euclidean` shaping. Оба алгоритма в одном сравнении используют **одну и ту же функцию**.
