# Act I educational animation plan

## Shared visual language

All scenes use a fixed 16:9 white canvas, the same map geometry, readable sans-serif typography, blue for ordinary motion, teal for a highlighted best trajectory, orange for the goal, gray for obstacles, blue/orange for genes inherited from parents A/B, and magenta for mutations. Checkpoint markers are hidden unless a scene explicitly teaches checkpoint-based reward design.

Introductory scenes use Euclidean goal-distance shaping because it supplies no obstacle-aware route. Sparse goal reward is used when the lesson is specifically about sparse feedback. Geodesic and checkpoint policies are reserved for the reward-design scene and labeled as additional map-specific knowledge.

## 1. Beetle genome to movement

- **Educational message:** A genome is a time-ordered sequence of relative turn commands; fixed physics converts commands into motion.
- **Required data:** One recorded genome, its deterministic rollout, collision events, world physics, and outcome metrics.
- **Composition:** Large map and moving beetle on the left; angle-versus-gene plot with the current command highlighted on the right.
- **Duration:** 6–8 seconds.
- **Labels:** Episode step, gene index, turn in degrees, blocked move, physical goal status, fixed step length.
- **Source:** A genome from a stored playback snapshot, replayed by the real simulator.
- **Priority:** Essential. Pilot scene 1.

## 2. Initial random population

- **Educational message:** Different independently sampled genomes explore different paths under identical physics.
- **Required data:** The complete first recorded random-search batch and all deterministic rollouts.
- **Composition:** Faint path histories with moving beetle markers on one map; a compact side panel states the shared conditions.
- **Duration:** 6–8 seconds.
- **Labels:** “Independent samples,” batch index, episode timestep, candidates still moving, physical successes.
- **Source:** Actual evaluated candidates from batch zero; never synthesized or described as offspring.
- **Priority:** Essential. Pilot scene 2.

## 3. Evaluating and selecting candidates

- **Educational message:** Physics produces outcomes, the chosen reward converts each outcome to a scalar, and selection uses only those scalars.
- **Required data:** One GA population snapshot, scores, shared outcome metrics, and selected parent identifiers from the operation log.
- **Composition:** Population map followed by a sorted score strip; selected candidates acquire outlines while geometry remains visually separated from the selection step.
- **Duration:** 7–9 seconds.
- **Labels:** Fitness policy, evaluations used, best-by-fitness versus physical success, selected parent IDs.
- **Source:** Recorded snapshot scores and bounded real GA operation events.
- **Priority:** Essential for the full catalog; not part of the pilot export.

## 4. Synthetic one-point crossover

- **Educational message:** One-point crossover combines control instructions, not connected pieces of geometric paths.
- **Required data:** Canonical deterministic eight-gene synthetic example, crossover cut, and two explicit mutations.
- **Composition:** Parent gene rows, cut marker, progressively assembled pre-mutation child, then magenta mutation highlights and final child.
- **Duration:** 7 seconds.
- **Labels:** Gene indices, degree values, cut index, mutation indices, “illustrative deterministic example.”
- **Source:** `synthetic_operation()`; radians internally and degrees only for display.
- **Priority:** Essential. Pilot scene 3.

## 5. Real crossover and mutation

- **Educational message:** The same operators occur in a real seeded GA run and can improve, preserve, or degrade fitness.
- **Required data:** One fully evaluated real operation event with parents, cut, pre-mutation child, mutations, final child, scores, and physical outcomes.
- **Composition:** Parent and child trajectories above a gene-operation strip; keep the map and fitness settings identical across all three.
- **Duration:** 8–10 seconds.
- **Labels:** Seed, generation, operation ID, parent/child scores, outcome metrics, event classification threshold.
- **Source:** Bounded GA operation log and deterministic replay. Missing classifications are reported rather than fabricated.
- **Priority:** Essential for the full catalog; not part of the pilot export.

## 6. Evolution across generations

- **Educational message:** Repeated evaluation and genetic variation can change the population and best-so-far trajectory across generations.
- **Required data:** Consecutive full GA snapshots, best-so-far genomes, evaluations, and actual candidate rollouts.
- **Composition:** Population movement within each generation, then an explicit transition card for selection/crossover/mutation before the next population appears.
- **Duration:** 15–20 seconds.
- **Labels:** Generation, episode timestep, cumulative evaluations, physical success, best-so-far fitness under one fixed policy.
- **Source:** Stored GA playback snapshots and operation metadata; rendering performs no optimization.
- **Priority:** Essential for the full catalog.

## 7. Random search versus local search

- **Educational message:** Random search samples independent batches; local search evaluates an incumbent and mutated neighbors and accepts only improvement.
- **Required data:** Matched random/local playback snapshots, local incumbent identity, neighbor scores, and acceptance decision.
- **Composition:** Two synchronized maps with method-specific annotations; random paths refresh independently while local incumbent remains visually persistent.
- **Duration:** 12–15 seconds.
- **Labels:** Batch index, episode timestep, evaluation count, “independent samples,” incumbent, neighbor, accepted/rejected.
- **Source:** Actual matched recorded batches and deterministic replay.
- **Priority:** Essential for the full catalog.

## 8. Synchronized GA/random/local comparison

- **Educational message:** Fair comparison requires identical physics, map, reward, seed identifier, population size, and evaluation budget.
- **Required data:** Matched cached playback for all three methods with identical evaluation coordinates.
- **Composition:** Three panels with shared scales and frame timing; best-so-far remains distinct from the moving current population.
- **Duration:** 15–20 seconds.
- **Labels:** Method, generation or batch, episode timestep, evaluations, physical success, best-so-far score.
- **Source:** Task 5 benchmark snapshots; terminal beetles remain at their recorded final positions.
- **Priority:** Essential for the full catalog.

## 9. Effect of fitness design on a maze or spiral

- **Educational message:** Reward design changes the information available to search; geodesic fields and ordered checkpoints add map-specific guidance.
- **Required data:** Matched seeds and budgets for sparse goal, Euclidean, geodesic, and optionally checkpoint policies on spiral or two-bridges.
- **Composition:** Synchronized example trajectories followed by reward-independent outcome summaries; no cross-policy raw-fitness comparison.
- **Duration:** 12–18 seconds.
- **Labels:** Reward information level, run success, final distance, collisions, path length, censored failures.
- **Source:** Actual fitness case-study runs. Failure remains visible when no optimizer reaches the goal.
- **Priority:** Optional extension after the core Act I story; scientifically important for discussion.
