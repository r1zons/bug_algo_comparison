# Task 4 — Interactive Jupyter playback of stored runs

Build a clean, presentation-ready Jupyter widget that replays **recorded** results from the core simulator. Do not embed the optimizer implementation in the notebook. Keep it usable offline after installing dependencies.

Controls:
- dropdown: map preset;
- dropdown: optimizer (`ga`, `local`, `random`);
- dropdown: fitness strategy;
- `IntSlider` for generation/batch index, with `Play` linked to the slider;
- seed and concise search settings;
- checkboxes: show all candidates, selected candidate, best-so-far, obstacles/checkpoints;
- button: run experiment (do not rerun when merely moving the slider).

Views: consistent 2D map, actual beetle movement/paths for a chosen generation, best-so-far route, `best_fitness vs evaluations` for a *fixed* fitness, and common success/step metrics. For local search show incumbent and neighboring candidates; for random search show independent fresh candidates; for GA show population. Label the horizontal axis as `evaluations` where it matters, not just `generation`.

Architecture and performance:
- Store bounded snapshots and replay deterministic candidate genotypes lazily when needed; avoid huge in-memory trajectory logs for every generation.
- Optional `ipywidgets` dependency with a helpful fallback if the notebook is run where widgets are unavailable. Matplotlib static renders must continue to work.
- A widget update should not rerun optimization or rebuild the geodesic map.
- Keep view/update logic in an importable `buglab` module; notebooks contain only initialization and brief guidance.

Acceptance criteria:
1. A user can play and scrub generations without rerunning optimization.
2. Changing map/fitness requires a new run or selecting a cached run; old data is never mislabeled.
3. Same run and seed yield identical visualizations on repeated playback.
4. Provide a screenshot or exported demo and run tests on selection/snapshot indexing.
