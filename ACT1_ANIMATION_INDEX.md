# Act I animation index

All files use a 1280×720 white canvas, Euclidean shaping, fixed simulator physics, and deterministic saved source data. Rendering replays saved genomes but never starts an optimizer. Checkpoint markers are hidden.

| Scene | GIF | Educational purpose | Map | Source | Lecture position |
|---|---|---|---|---|---|
| 01 | `01_beetle_model.gif` | A genome is a sequence of relative turn commands. | `open_field` | Actual random-search candidate, seed 12 | Introduce the beetle model |
| 02 | `02_population.gif` | Different genomes produce different paths. | `open_field` | Actual independent random batch, seed 12 | Introduce population diversity |
| 03 | `03_evaluation_selection.gif` | Simulate, score, then select likely parents. | `open_field` | Actual traced GA population and tournament selections, seed 12 | Explain evaluation and selection |
| 04 | `04_crossover_mutation.gif` | Crossover combines command sequences; mutation changes genes. | Illustrative gene diagram | Deterministic synthetic example | Explain genetic operators |
| 05 | `05_one_generation_transition.gif` | Inspect one recorded generation update in slow motion. | `default` two-obstacle map | Actual traced GA generations 0→1, seed 0 | Put the GA loop together |
| 06 | `06_simple_evolution.gif` | Repeated updates improve the best-so-far path. | `default` two-obstacle map | Actual 12-generation GA run, seed 0 | Show evolution at overview speed |
| 07 | `07_two_bridges_evolution.gif` | A richer population can sample upper and lower route families. | `two_bridges` | Actual 14-generation GA run, seed 4 | Introduce multiple useful routes |
| 08 | `08_method_comparison.gif` | GA, random search, and local search explore differently. | `two_bridges` | Matched actual runs, seed 4, 224 evaluations each | Compare search semantics |
| 09 | `09_spiral_fitness_problem.gif` | Straight-line proximity can reward the wrong side of a wall. | `spiral` | Actual Euclidean-fitness GA run, seed 0; physics-valid corridor reference for explanation | Motivate fitness design |

## Reproduction

Optimization is an explicit preparation step:

```bash
python -m buglab.act1 --output examples/act1
```

Animation export uses only those saved caches and operation logs:

```bash
python -m buglab.scenes \
  --manifest act1_scenes.json \
  --output outputs/act1
```

The declarative source and render settings are in `act1_scenes.json`. Source hashes and run outcomes are in `examples/act1/DATA_INDEX.json`. Every GIF has a same-named JSON provenance sidecar.

## Limitations and teaching simplifications

- Scene 04 is explicitly synthetic; all movement in the other scenes comes from actual simulator trajectories.
- Scene 09 overlays a deterministic, physics-valid checkpoint-following corridor solely to explain geometry. It was not available to the Euclidean-fitness optimizer.
- Euclidean shaping is intentionally presented as a simple introductory rule, not a generally sufficient objective.
- Scene 07 shows the route families that occur in the selected deterministic run. Most later candidates use the lower passage; the visualization does not manufacture balanced diversity.
- Scene 08 is one matched illustrative run, not statistical evidence that one method is superior.
