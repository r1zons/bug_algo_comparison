# Codex task — Final Act I visual polish and slide-ready export

Continue in `r1zons/bug_algo_comparison`, starting from commit `1494ee5` or a newer reviewed commit.

Read `AGENTS.md`, `ARCHITECTURE.md`, `ANIMATION_PLAN.md`, `ACT1_ANIMATION_INDEX.md`, `act1_scenes.json`, and the relevant code under `buglab/`. Inspect the existing generated GIFs and provenance before editing.

## Objective

Finish the visuals needed to assemble Act I of a university lecture. Prioritize teaching clarity, correct scientific interpretation, and reliable slide integration. **Do not** build a new UI, add optimizers, change beetle physics, or rewrite the architecture.

## 1. Replace scene 04 with static slides

Retire `04_crossover_mutation.gif` from the **main lecture sequence** (retain the historical output separately). Produce four PNG/SVG slides:

1. Two illustrative parent genomes and their physically simulated paths.
2. One-point crossover: child inherits instructions 0–3 from A and 4–7 from B; display the simulated child trajectory (not a stitched path).
3. Change one turn gene and show the trajectory *before and after* mutation.
4. Comparison of all four trajectories, explaining that mutation/crossover do not guarantee improvement.

A standalone, self-contained example has already been prepared and can be imported from the provided `act1_crossover_static.zip`. If that archive is unavailable in the Codex workspace, reproduce it from the following verified specification:

- Start `(12, 32)`, goal `(76, 33)`; `8` time steps, `8.0` units per step; max relative turn `24°`.
- Initial direction points from start to goal. At each time step, rotate by the indicated relative angle **then** move a fixed distance.
- Parent A in degrees: `[6, 10, -2, -8, 4, 4, -6, -5]`.
- Parent B in degrees: `[-6, -10, 4, 10, -2, -4, 6, 5]`.
- Cut after index `3` (zero-based; first four genes from A).
- Crossover child: `[6, 10, -2, -8, -2, -4, 6, 5]`.
- Mutated child: `[6, 10, -2, -8, -2, -22, 6, 5]` (one mutation at zero-based index 5).

Use Russian slide copy but keep source code, variable names, and developer documentation in English. Distinguish **illustrative data** from a real GA event. Do not claim these paths were generated in scene 05.

## 2. Clarify Euclidean rewards and repair the spiral explanation

The existing `euclidean` policy scores `-min_t distance(position_t, goal)`, i.e. the best approach **at any point**, not final distance. Preserve its semantics and historical outputs.

- Correct any narration or labels calling this policy "final distance" or "final performance".
- Introduce a separate backward-compatible named policy, e.g. `final_euclidean`, if needed for the spiral demonstration: `score = -norm(trace.positions[-1] - goal)`.
- Test the new policy explicitly, including the difference from closest-approach Euclidean scoring.
- Use the correct policy name in every figure, cache key, provenance JSON, and slide caption.
- In the spiral demonstration, explicitly show a physically valid detour whose distance to the goal **initially increases**. Distinguish geometric reference trajectories from GA-discovered solutions. The reference must be validated against the simulator collision rules; if it does not reach the goal, label it as an **incomplete** reference route, never a solved route.
- Do not pretend that GA failure on one seed proves reward misspecification; explain that the example illustrates a source of misleading selection pressure.
- Keep geodesic/checkpoint information out of the introductory reward, and keep checkpoint markers hidden by default.

## 3. Improve main GIFs for lecture use

Focus on `05_one_generation_transition`, `06_simple_evolution`, `07_two_bridges_evolution`, and `08_method_comparison`.

- Use clean 16:9 white slide graphics and consistent colors; larger field, fewer diagnostics, readable labels at 1280×720.
- Increase animation fluidity to approximately 12–15 fps, with substantially more sampled/interpolated frames per beetle episode than the current 6 fps exports; visual interpolation between actual adjacent physical positions is allowed but may not change the simulated trajectory or optimizer behavior.
- Keep the detailed generation transition (scene 05) **before** the overview (scene 06).
- Make the selection/offspring phases explicit without confusing one particular crossover event with the entire next population.
- Persistently distinguish the current generation's candidates from **best-so-far**.
- Make local search's incumbent/neighbors and acceptance decisions visibly different from the independent random batches and GA offspring.
- Do not use raw reward values for comparing different fitness policies.
- Consider MP4 alongside GIF if files become too heavy; GIF must remain available.
- Add small final-frame holds where helpful for lecture commentary.

## 4. Verify the two-bridges route classification

The current family detector samples locations near the central strip. Check crossing using the *actual crossing segment of the barrier's central x-coordinate*, and classify only physically valid passages. Do not infer a successful end-to-end route merely from crossing through a bridge. Update legend labels or explanatory wording if necessary.

## 5. Final export bundle

Deliver a clean `outputs/act1_final/` directory with:

- 01–03 polished GIFs or retained originals if they pass visual review.
- 04a–04d static crossover PNGs (and SVGs if practical), not a mandatory GIF.
- 05–09 polished GIFs (and optional MP4).
- Provenance JSON sidecars, a complete index, and short Russian presenter notes (1–3 sentences per scene).
- A four-column contact sheet showing first, middle, final frames and the message of each animated scene.

Also prepare a short `ACT1_SLIDE_ASSEMBLY.md` mapping assets into a **13–16 slide** Act I sequence: introduce model and population; simple fitness and probabilistic selection; static synthetic crossover/mutation; one-generation transition; multi-generation evolution; general GA diagram; two bridges; comparison with random/local search; brief examples of other optimization tasks (TSP, scheduling, neural-network parameter search); encoding and admissibility; the spiral fitness failure; realistic limitations; and a transition toward evolving neural-network controllers.

Keep theoretical subtleties (selection probabilities, stochastic kernels, convergence, advanced crossover variants) briefly named and linked to appendix slides rather than unpacked in the introductory narrative.

## Validation

- Run full pytest suite and historical CLI regression; report any intentional metric naming changes.
- Verify scene 04 and scene 05 are explicitly described as *different examples*.
- Test all provenance hashes and deterministic reproduction for newly exported assets.
- Render every target GIF; inspect beginning, middle, and last frames. Check labels, clipping, timing, color continuity, and physical plausibility.
- Do not manufacture trajectories or replace observations with desirable fake outcomes.
- Do not merge to `main` or delete old files without review. Work in a separate branch and provide changed files, tests, contact sheet, and limitations.

Stop after final Act I assets and slide assembly notes. Remember that the best-so-far curve is monotone by definition; it does not imply that every new individual or generation improved.

The next project phase is to **assemble the actual Act I presentation**, then proceed to Acts II–IV without further expanding this demo simulator.
