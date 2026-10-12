# Act I focused refinement index

This additive package explains how the three existing optimizers generate candidates and replaces the historical square-spiral lecture case with a certified elongated rectangular corridor. It does not overwrite `outputs/act1_final/` or change physics, optimizer rules, or legacy datasets.

## Lecture order

| Order | Asset | Purpose | Recorded source |
|---:|---|---|---|
| A1 | `outputs/act1_refined/A1_random_independent_attempts.gif` | Independent complete Random Search genomes; best-so-far is reporting only. | `examples/act1/bridges_random.npz`, seed 4 |
| A2 | `outputs/act1_refined/A2_local_incumbent_neighbors.gif` | Complete incumbent and neighbor episodes, followed by accept/reject between episodes. | `examples/act1/bridges_local.npz`, seed 4 |
| A3 | `outputs/act1_refined/A3_ga_population_variation.gif` | Population evaluation, real parent selection, crossover, mutation, and offspring evaluation. | `bridges_ga.npz` plus the bounded operation log, seed 4 |
| A4 | `outputs/act1_refined/A4_matched_method_comparison.gif` | Concise comparison after the mechanisms are established. | All three Two Bridges records, seed 4, 224 evaluations each |
| B2 | `outputs/act1_refined/B2_rectangular_spiral_optimizer.gif` | Pose the reward-design problem without revealing a route. | Actual 224-evaluation GA run, seed 0 |
| B1 | `outputs/act1_refined/B1_rectangular_spiral_barrier.gif` | Reveal the all-path distance barrier and one physics-valid route. | Analytic certificate plus deterministic simulator reference |

Each GIF has a 1920×1080 PNG poster and JSON sidecar in the same directory. `ACT1_REFINED_CONTACT_SHEET.png` contains the first, middle, and final frame of every scene.

## Reproduction

Optimization is an explicit preparation step:

```bash
python -m buglab.refined_data --output examples/act1_refined
```

Rendering consumes only saved optimizer records:

```bash
python -m buglab.refined_scenes \
  --manifest act1_refined_scenes.json \
  --output outputs/act1_refined
```

The declarative manifest fixes duration, frame rate, selected batches or generations, candidate count, source paths, and educational message. Sidecars record hashes of source and exported artifacts and state `rendering_launches_optimizer: false`.

## Scientific interpretation

- Random Search samples complete open-loop genomes independently from a clipped normal distribution. Its paths can look smooth because turn perturbations are small and the simulator initially points toward the goal; this is not hidden navigation logic.
- Local Search mutates a complete incumbent genome, simulates every neighbor as a complete episode, then accepts a strict fitness improvement. It is not an online greedy controller.
- GA alone uses a population with tournament selection, one-point crossover, mutation, and elitism. A recorded operation is shown, not a synthetic success.
- The A4 comparison controls map, physics, fitness, population, seed identifier, and evaluation budget. A single seed illustrates mechanisms and is not evidence of general optimizer superiority.
- On `rectangular_spiral`, the radius-inflated lower wall is attached to the left boundary. Therefore every collision-free center path from start to goal must pass its right edge, at least 47.1 units from the goal versus 15.3 initially: a certified positive barrier of 31.8 units.
- `euclidean` remains the historical negative minimum distance over the full episode. It can retain an earlier close approach and plateau during a required detour. `final_euclidean` uses the terminal distance and can worsen while the beetle moves away. Neither observation proves that GA can never reach the goal.
- The B2 run did not physically reach the goal. B1's reference genome does reach it in the unchanged simulator, but is post-hoc explanatory material and was never exposed to the optimizer or reward.

## Presentation simplifications

Only the highest-scoring local neighbor and a small subset of surrounding trajectories receive strong emphasis. Genetic operators are summarized at the operation level instead of displaying all 90 gene values. Short Russian captions carry the essential on-frame message; configuration detail belongs in the sidecars and lecture notes.
