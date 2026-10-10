# Task 5 — Fair comparison, map gallery, and exportable teaching media

Extend visualization and experiments without changing optimizer rules. Export matched visualizations of GA, random search, and local search on the same map/fitness/seed and budget. All three should share the same map coordinates, obstacles, playback cadence and labels.

Required exports:
- beetles-moving GIF or MP4 per method, showing **real** stored/replayed trajectories;
- one side-by-side comparison clip or an exportable figure;
- slider-readable snapshots for arbitrary recorded generation/batch;
- best-so-far vs **fitness evaluations** plot with repeated seeds, median and IQR;
- success rate, evaluations-to-first-success, collisions, steps-to-goal and path length as independent metrics;
- comparative Euclidean/geodesic/checkpoint shaping study on spiral and two bridges, evaluated by shared outcome metrics, not directly compared reward values.

Budget fairness:
- Keep objective function and map identical when ranking optimizers.
- Count every evaluated candidate, including initial populations and elites if re-evaluated; avoid double counting when cached (document policy).
- Report environment-step cost as a supplementary metric because candidates may terminate early.
- Do not claim algorithm superiority from 3 seeds; provide configurable 20–30 seed benchmark and label quick demos as demos.
- Distinguish independent random search from local search (an incumbent plus mutations) and GA (selection/crossover/population).

Acceptance criteria:
1. Headless test run can regenerate artifacts from one command with fixed seeds.
2. At least one demo export for each method looks legible in 16:9 slides on white background.
3. CSV/JSON includes all map, fitness, optimizer configs and reproducible seed identifiers.
4. Plot values are generated from actual runs; all figures state method, map, budget and metric.
5. Include a short interpretation explaining what can and cannot be concluded.
