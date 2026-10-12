# Act I slide assembly guide

This 15-slide structure keeps one teaching claim per slide and uses the polished assets in `outputs/act1_final/`. It is an assembly plan, not a PowerPoint file.

| Slide | Teaching purpose | Primary asset | Speaker emphasis |
|---:|---|---|---|
| 1 | Introduce genotype-to-phenotype mapping | `01_beetle_model.gif` | Relative turn commands become motion under fixed physics. |
| 2 | Introduce population diversity | `02_population.gif` | Same world and physics, different genomes. |
| 3 | Introduce simple fitness and probabilistic selection | `03_evaluation_selection.gif` | `fitness = -min_t Euclidean distance`; better means more likely, not certain. |
| 4 | Establish two parents | `04a_crossover_parents.png` | Commands are programs; paths are simulator outcomes. |
| 5 | Explain one-point crossover | `04b_crossover_child.png` | Copy genes 0–3 from A and 4–7 from B; never describe this as stitching paths. |
| 6 | Explain mutation and evaluation | `04c_crossover_mutation.png`, then `04d_crossover_outcome.png` | A marked command changes; the result is not guaranteed to improve. |
| 7 | Inspect one complete generation update | `05_one_generation_transition.gif` | One logged child illustrates the operation; the full next population is evaluated. |
| 8 | Summarize the general GA loop | Still from Scene 05 plus a simple slide-native loop: initialize → evaluate → select → vary → repeat | Name selection probabilities, stochastic kernels, and crossover variants only as appendix topics. |
| 9 | Show repeated evolution | `06_simple_evolution.gif` | Separate the current population from best-so-far; monotone best-so-far is bookkeeping. |
| 10 | Introduce a richer route landscape | `07_two_bridges_evolution.gif` | Barrier-opening classification is not a goal-success claim. |
| 11 | Compare three search semantics | `08_method_comparison.gif` | GA population, independent random batches, local incumbent plus neighbors; matched budget. |
| 12 | Generalize beyond beetles | Three compact slide-native examples: TSP tours, scheduling permutations, neural-network parameter vectors | Encoding determines what variation means. No new simulator result is implied. |
| 13 | Discuss encoding and admissibility | Reuse the gene rows from `04b_crossover_child.svg` | Bounds and repair keep commands admissible; physics still decides whether motion is blocked. |
| 14 | Reveal the fitness-design limitation | `09_spiral_fitness_problem.gif` | Closest approach can reward the wrong side of a wall; the valid detour injects explanatory route knowledge. |
| 15 | State realistic limitations and transition to Acts II–IV | Final still from Scene 09 plus a short limitations list | One seeded run is illustrative, fitness design matters, and the next question is evolving richer controllers such as neural networks. |

## Appendix pointers

Keep selection probability formulas, convergence qualifications, stochastic transition kernels, elitism details, and advanced crossover variants in appendix slides. The main narrative needs only the operational meanings shown by the assets.

## Integration notes

- Use GIF playback once per slide and hold the last frame for discussion.
- Prefer the SVG versions of Scene 04 when the presentation editor supports them.
- Do not label a Two Bridges crossing as success unless the recorded physical `reached` outcome is true.
- Keep the Scene 09 formula visible: the selected policy is minimum-over-trajectory distance, not terminal distance.
- The original `04_crossover_mutation.gif` is optional backup material under `outputs/act1_final/optional/`; it is not part of the main sequence.
