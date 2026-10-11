# Benchmarking and interpretation

## Commands

Fast deterministic smoke benchmark:

```bash
python -m buglab.benchmark --preset smoke --output outputs/task5_smoke
```

Complete headless lecture export, including animations and the fitness-design case study:

```bash
python -m buglab.benchmark --preset smoke --media --case-study --output outputs/task5_lecture
```

The statistically more useful preset has 25 seeds and is deliberately opt-in:

```bash
python -m buglab.benchmark --preset full --output outputs/task5_full
```

## Metric definitions

- **Run success rate** is the fraction of seeds where at least one genome physically reaches the goal.
- **Evaluated-genome success fraction** is the fraction of submitted candidates that reach the goal. It can count repeated GA elites and local-search incumbents and is not an independent-sample success rate.
- **First-success evaluations** is the exact one-based candidate ordinal. Failed runs are right-censored at their evaluation budget. Summaries report the Kaplan–Meier median when it is identifiable and the number censored.
- **Best-so-far fitness** is summarized with median and IQR only for runs using the same fitness policy.
- Final distance, steps to goal, collisions, path length, and environment steps are physical outcomes independent of reward scale. Successful-run route metrics describe the fastest successful trajectory; otherwise they describe the best-by-fitness failure and are labeled accordingly.

Every submitted candidate consumes one evaluation. This includes a GA elite when it is submitted again and candidate zero when local search resubmits its incumbent. Cached playback and rendering do not add optimizer evaluations.

## Legitimate conclusions and limitations

The smoke preset verifies reproducibility, accounting, and the export pipeline; three seeds cannot establish optimizer superiority. The 25-seed preset gives more stable medians and IQRs but still applies only to these maps, open-loop turn genomes, hyperparameters, and budgets.

Euclidean, geodesic, and checkpoint rewards have different numerical scales, so their raw fitness values are never compared. Geodesic distance supplies an obstacle-aware distance field and checkpoints supply a designer-authored ordered corridor. Both inject map-specific information unavailable to Euclidean distance. If a configuration has no successful run, that failure remains censored and visible rather than being tuned away.

Animations are explanatory views of actual evaluated or deterministically replayed genomes. They distinguish optimizer iteration from episode timestep, hold early terminal positions, and do not imply crossover for random search or inheritance for local-search neighbors.
