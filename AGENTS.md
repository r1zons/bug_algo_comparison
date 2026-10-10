# Bug Evolution Lab — coding agent guidance

This repository is a reproducible teaching simulator for comparing optimizers on beetle path-planning tasks. Preserve its educational clarity and numerical integrity.

## Repository map
- `buglab/environment.py`: beetle motion, collisions, grid distance field, rollout. Read for world/map/scoring work.
- `buglab/optimizers.py`: `ask()`/`tell()` optimizers (GA, local search, random search). Read for operator changes.
- `buglab/experiment.py`: experiment execution, logging and comparisons.
- `buglab/visualization.py`: PNG and GIF rendering.
- `tests/`: deterministic tests.
- `ARCHITECTURE.md`: existing contracts and caveats.

## Invariants
- A genotype is a sequence of **relative turn angles in radians**, with fixed-length steps and a finite episode horizon. Angles are clamped to ±`max_turn`.
- A beetle never passes through obstacles or field boundaries; a blocked move consumes the step but the heading still changes.
- Physics is the same for every optimizer. Optimizers must not access geometry, trajectories, geodesic maps, or reward internals; they only receive scalar fitness via `ask()` / `tell()`.
- Every search method gets a comparable count of evaluations. Record **evaluation count**, not just generation number; separate environment-step cost when relevant.
- Fitness is a *training signal*. Across different fitness functions, compare methods using common outcome metrics (success rate, time-to-goal, collisions, traveled distance, evaluations-to-success), not raw fitness values.
- Geodesic shaping uses an obstacle-aware shortest-path distance grid, which supplies additional problem knowledge. Label it clearly; do not treat it as equivalent to Euclidean shaping.
- Use deterministic NumPy generators with explicit seeds. No invented success rates or fabricated algorithm trajectories.

## Engineering guidance
- Improve the existing project incrementally. Do not rewrite it or introduce a general RL framework.
- Prefer small pure functions, typed dataclasses, explicit config validation and lightweight dependencies. Separate maps, physics, fitness, search, logging and UI.
- New code, comments, docstrings, file names and developer documentation must be in English.
- Add regression and edge-case tests for any algorithm/physics changes; preserve the existing command-line entry point and notebook or provide clearly documented compatibility wrappers.
- Avoid notebooks containing the only copy of business logic. Put logic in importable modules; notebooks remain thin clients.
- Generated outputs, caches, GIFs and run logs should not be committed unless intentionally added as small pedagogical examples.

## Commands
```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m buglab --generations 8 --population 16 --seeds 0 --methods ga random local
```

The local tests are disposable and safe. Run the relevant tests, fix regressions, and rerun without requesting approval for routine edits. Report unverified assumptions rather than silently changing experiment semantics.
