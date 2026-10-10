# Codex implementation handoff

The goal is to upgrade the existing beetle path-planning laboratory for an educational talk, not to build a new optimization platform. Read `AGENTS.md` and `ARCHITECTURE.md`. Start with `codex_tasks/01_refactor.md`. Do one task at a time in a separate branch or checkpoint; run tests and visually inspect relevant outputs before starting the next task.

## Order of work
1. `01_refactor.md` — safety-first modularization and regression baseline.
2. `02_maps_and_fitness.md` — five maps, explicit fitness variants, common metrics.
3. `03_operator_story.md` — simple + real crossover trace, lineage.
4. `04_jupyter_player.md` — Jupyter generation slider/play with map/algorithm selectors.
5. `05_benchmarks_and_exports.md` — same-map GA/local/random animated exports and fair reports.
6. `06_optional_ui.md` — only if the previous five tasks pass and the core is stable.

## Workflow
In Codex, open the repository and paste the **task prompt** from the relevant file; do not paste all six as a single mega-task. Ask Codex to finish a task, run the tests, provide a concise summary and screenshots/plots where possible. Review the diff before the next task. Keep successful stages as separate commits/PRs. Do not merge a partially passing stage.

## Existing code facts (verified before handoff)
- `python -m pytest -q`: 3 passed.
- `World` currently bundles geometry and fitness; two fitness modes exist (`geodesic`, `euclidean`).
- `run_method` records snapshots but not an inspectable full-generation history.
- `GeneticAlgorithm.example_operation` preserves a single pair/cut/mutation example; more are needed.
- Local and random search share the scalar-only `ask()/tell()` interface.
- The existing README contains Russian text; new developer docs/code should use English. Do not rewrite historical results.

## Desired deliverables
- Runnable and tested importable Python modules.
- Notebook with generation slider and `Play`, including a static fallback.
- Exportable GIF or MP4 with beetles moving; representative crossover and baseline-method visualizations.
- Experiments with clearly distinguished algorithm performance metrics and fitness-design effects.
- A small gallery of 5 maps with tests for reachability/collision logic.

**Scope rule:** Future neural-network policies and reinforcement learning are **not part of these tasks**. Keep the system extensible, but do not implement them now.
