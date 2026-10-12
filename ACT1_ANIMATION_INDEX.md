# Act I visual-polish index

Final assets are in `outputs/act1_final/`. Animated scenes use a 1280×720 white canvas at 12 fps, fixed simulator physics, and deterministic saved source data. Rendering replays saved genomes and never starts an optimizer. Checkpoint markers are hidden. Scene 04 is now a lecturer-paced static sequence; its historical GIF is retained under `optional/`.

| Scene | Final asset | Educational purpose | Map | Optimizer / source | Lecture position |
|---|---|---|---|---|---|
| 01 | `01_beetle_model.gif` | A genome is a sequence of relative turn commands. | `open_field` | Actual random-search candidate, seed 12 | Introduce the beetle model |
| 02 | `02_population.gif` | Different genomes produce different paths. | `open_field` | Actual independent random batch, seed 12 | Introduce population diversity |
| 03 | `03_evaluation_selection.gif` | Score closest approach, then probabilistically select parents. | `open_field` | Actual traced GA population and tournament selections, seed 12 | Explain evaluation and selection |
| 04a | `04a_crossover_parents.png` / `.svg` | Two parent command sequences produce different simulated paths. | Illustrative empty field | Deterministic synthetic example | Introduce parents |
| 04b | `04b_crossover_child.png` / `.svg` | One-point crossover combines commands, not trajectory pieces. | Illustrative empty field | Deterministic synthetic example | Explain crossover |
| 04c | `04c_crossover_mutation.png` / `.svg` | One marked gene changes from −4° to −22°. | Illustrative empty field | Deterministic synthetic example | Explain mutation |
| 04d | `04d_crossover_outcome.png` / `.svg` | A new child must still be evaluated; improvement is not guaranteed. | Illustrative empty field | Deterministic synthetic example | Close the operator sequence |
| 05 | `05_one_generation_transition.gif` | Inspect one recorded population update without treating one child as the whole next generation. | `default` two-obstacle map | Actual traced GA generations 0→1, seed 0 | Put the GA loop together |
| 06 | `06_simple_evolution.gif` | Repeated recorded generations update the population while best-so-far remains distinct. | `default` two-obstacle map | Actual 12-generation GA run, seed 0 | Show evolution at overview speed |
| 07 | `07_two_bridges_evolution.gif` | Candidates can cross either physical opening; crossing does not imply success. | `two_bridges` | Actual 14-generation GA run, seed 4 | Introduce multiple route families |
| 08 | `08_method_comparison.gif` | GA populations, independent random batches, and local neighbors have different semantics. | `two_bridges` | Matched actual runs, seed 4, 224 evaluations each | Compare search strategies |
| 09 | `09_spiral_fitness_problem.gif` | Closest-approach fitness can favor a wall-separated point and disfavor a necessary detour. | `spiral` | Actual closest-approach Euclidean GA run, seed 0; separate physics-valid reference | Motivate fitness design |

## Reproduction

The existing source experiments are recorded under `examples/act1/`. Recreating them is an explicit, separate action:

```bash
python -m buglab.act1 --output examples/act1
```

The final export reads only those caches and operation logs:

```bash
python -m buglab.act1_bundle \
  --manifest act1_final_scenes.json \
  --output outputs/act1_final
```

To rebuild only the visual inspection sheet from already-rendered GIFs:

```bash
python -m buglab.act1_bundle \
  --manifest act1_final_scenes.json \
  --output outputs/act1_final \
  --contact-sheet-only
```

Each final GIF has a same-named JSON sidecar containing scene configuration, saved source paths, renderer settings, and SHA-256 artifact hash. The static frames have sidecars containing the exact radian genomes, displayed degree values, operation indices, simulator outcome, and PNG/SVG hashes.

## Limitations and teaching simplifications

- Scenes 01–03 and 05–09 replay actual evaluated candidates. Scene 04 is explicitly synthetic, although every trajectory shown in it is produced by the unchanged simulator from the displayed genome.
- `euclidean` means `-min_t distance(position_t, goal)`, not terminal distance. The separate `final_euclidean` policy exists to prevent that distinction from being hidden.
- Scene 07 colors only verified crossings of the central barrier line through an open gap. The color is not a success label; physical arrival at the goal is recorded separately.
- Scene 08 is one matched illustrative run, not statistical evidence that one method is generally superior.
- Scene 09 uses a deterministic, collision-free checkpoint-following reference that reaches the goal. It is explanatory domain knowledge and was unavailable to the closest-approach optimizer. One failed seeded run demonstrates a plausible selection pressure, not a proof that GA cannot solve the map.
- The animation interpolates only between adjacent recorded physical positions. It does not alter collisions, optimizer behavior, genomes, fitness values, or episode outcomes.
- Best-so-far is monotone by definition; it does not imply that every new individual or every generation improved.
