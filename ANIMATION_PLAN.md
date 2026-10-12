# Act I educational animation plan

## Shared visual language

Act I uses a fixed 16:9 white canvas, stable map coordinates, large sans-serif labels, blue for ordinary trajectories, teal for best-so-far or selected trajectories, orange for the goal and Euclidean guidance, gray for obstacles, purple for an alternative bridge route, blue/orange for inherited genes, and magenta for mutation or distance-increasing corridor segments. Checkpoint markers are hidden.

Closest-approach Euclidean fitness (`-min_t distance`) is the deliberately simple introductory policy. It is easy to explain and useful on the early maps, but it contains no obstacle-aware routing information and is not terminal-position distance. Scene 09 exposes that limitation rather than introducing a more sophisticated reward as a solution.

## Block A — The beetle model

### 01. A genome becomes movement — essential

- **Message:** A genome is a sequence of relative turn commands that produces a trajectory.
- **Data:** One actual candidate from saved open-field random search, seed 12.
- **Composition:** One large empty field, growing trajectory, moving beetle, compact command label.
- **Duration:** 6 seconds.
- **Labels:** Command index, angle in degrees, executed steps, physical goal status.
- **Source:** `examples/act1/open_random.npz`; deterministic simulator replay.

### 02. Different genomes explore different paths — essential

- **Message:** Different genomes produce different trajectories under identical physics.
- **Data:** One actual 12-candidate independent random batch.
- **Composition:** Simultaneous faint paths and moving markers on the same empty field.
- **Duration:** 5.5 seconds.
- **Labels:** Episode step, number of genomes, physical successes, “no crossover yet.”
- **Source:** `examples/act1/open_random.npz`.

### 03. Evaluate, then select parents — essential

- **Message:** After simulation, better evaluated candidates are more likely to become parents.
- **Data:** Actual GA generation-zero paths, Euclidean scores, and traced tournament parent use.
- **Composition:** Population map plus a compact score strip; actual frequently selected parents turn teal.
- **Duration:** 5.5 seconds.
- **Labels:** Simulate, score the closest approach over the whole trajectory, select; “higher is better.”
- **Source:** `open_ga.npz` and `open_ga_operations.json`, seed 12.

### 04. Crossover combines programs; mutation changes commands — essential static sequence

- **Message:** New genomes combine parent command sequences and then receive mutations.
- **Data:** Canonical deterministic eight-gene synthetic operation.
- **Composition:** Four 16:9 PNG/SVG frames: parents, one-point child, explicit mutation, then all simulated trajectories.
- **Duration:** Lecturer-paced static sequence; the original seven-second GIF remains optional.
- **Labels:** Degree values, cut index, mutation arrows, “illustrative deterministic example.”
- **Source:** `synthetic_operation()`; radians internally, degrees for display.

## Block B — Evolution on a simple obstacle map

### 05. One generation update, step by step — essential

- **Message:** One update evaluates a population, selects parents, creates offspring, and evaluates the next generation.
- **Data:** Actual traced generations 0 and 1 on the two-obstacle `default` map.
- **Composition:** Generation-zero movement, selected parent highlights, a short operator transition, then generation-one movement with the recorded offspring highlighted.
- **Duration:** 9 seconds.
- **Labels:** Generation, selection, crossover cut, mutation count, recorded offspring.
- **Source:** `simple_ga.npz` and `simple_ga_operations.json`, seed 0.

### 06. Repeated generations improve the search — essential

- **Message:** Repeating the update can gradually improve the population and best-so-far path.
- **Data:** Actual 12-generation GA run on the same map.
- **Composition:** Faint current population with a persistent teal best-so-far trajectory.
- **Duration:** 12 seconds.
- **Labels:** Generation, evaluation count, fixed-policy best fitness.
- **Source:** `simple_ga.npz`, seed 0.

## Block C — Multiple routes and search strategies

### 07. A population can explore different route families — essential

- **Message:** A richer map can admit qualitatively different useful routes.
- **Data:** Actual 14-generation GA run on `two_bridges` selected for naturally occurring upper and lower crossings.
- **Composition:** Blue lower-passage paths, purple upper-passage paths, gray unclassified paths, teal best-so-far.
- **Duration:** 10 seconds.
- **Labels:** Barrier-crossing legend, generation, evaluations, best fitness, and an explicit warning that crossing does not imply goal success.
- **Source:** `bridges_ga.npz`, seed 4. The observed imbalance between route families is preserved.

### 08. Three search strategies, one shared problem — essential

- **Message:** GA, random search, and local search explore the same problem differently.
- **Data:** Actual matched runs with the same map, Euclidean fitness, seed 4, population 16, and 224 evaluations.
- **Composition:** Synchronized GA, random, and local panels with shared axes and timing.
- **Duration:** 12 seconds.
- **Labels:** Population + inheritance; independent new batch; incumbent and neighbor acceptance; generation/batch and evaluations.
- **Source:** `bridges_ga.npz`, `bridges_random.npz`, and `bridges_local.npz`.

## Block D — Fitness-design limitation

### 09. When closer to the goal is not better — essential

- **Message:** Minimum-over-trajectory straight-line distance can reward one unreachable close approach through a wall.
- **Data:** Actual Euclidean-fitness GA candidates on `spiral`, plus a separately labeled physics-valid corridor reference.
- **Composition:** One spiral map, Euclidean distance rings and direct line, orange optimizer paths, then a dashed feasible corridor with distance-increasing segments in magenta.
- **Duration:** 7 seconds.
- **Labels:** The exact `-min_t distance` definition, “across a wall,” a physically valid distance-increasing detour segment, and an annotation at the misleading near-goal point.
- **Source:** `spiral_ga.npz`, seed 0. The explanatory corridor follows map checkpoints but was not available to the optimizer.

## Production boundary

`python -m buglab.act1` explicitly creates the seven source runs. `python -m buglab.act1_bundle --manifest act1_final_scenes.json --output outputs/act1_final` loads only saved data and exports the polished animations, static Scene 04 sequence, provenance sidecars, and inspection contact sheet. Rendering never launches optimization. `act1_scenes.json` and `outputs/act1_before_polish/` remain available for historical comparison.

## Focused method and distance-barrier refinement

The additive package in `outputs/act1_refined/` leaves the reviewed Act I assets intact. Its method sequence makes the candidate-generation mechanisms explicit before the matched comparison:

1. Random Search: every shown attempt is a newly sampled complete genome; the dashed best-so-far path is reporting state only. Small relative-turn variance and an initial heading toward the goal explain naturally smooth-looking paths.
2. Local Search: candidate 0 remains the incumbent while complete mutated neighbors run independent episodes; a strict improvement is accepted only between episodes. The selected sequence contains natural acceptances and rejections and is not online navigation.
3. GA: a real traced update shows an evaluated population, selected parents, crossover and mutation, and the resulting candidate's complete episode.
4. Matched comparison: all three recorded runs use `two_bridges`, Euclidean fitness, seed 4, population 16, and 224 evaluations. It illustrates mechanics for one seed, not superiority.

The new `rectangular_spiral` case separates the problem from its reveal. First show the recorded Euclidean-fitness optimizer run (`B2`); then reveal the analytic separator certificate and physics-valid route (`B1`). The route is post-hoc explanatory data, never reward input. The graphic distinguishes closest-approach `euclidean`, whose remembered minimum can plateau during a detour, from `final_euclidean`, which decreases as the current terminal point moves away.
