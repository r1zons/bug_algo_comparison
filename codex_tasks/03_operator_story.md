# Task 3 — Explain crossover and mutation with reproducible examples

Add a teaching-friendly operator trace without contaminating the experiment core.

First implement a **synthetic, fully deterministic** six-to-twelve gene example. Display parent A, parent B, cut point, pre-mutation child, changed gene indices/angles, final child and a one-sentence explanation that crossover combines control *instructions*, not geometrically connected path segments.

Then record **real** GA parent/offspring events from runs. Extend current `GeneticAlgorithm.example_operation` (currently one event only) into an optional, bounded log containing generation/seed, source parent IDs and scores, crossover cut, pre-mutation child, mutation index and value deltas, post-mutation child, and evaluated child score. Maintain correct parent/child alignment with `ask()/tell()` and handle elitism explicitly. Avoid storing full trajectories for every gene event by default: replay with `evaluate()` only when selected for rendering.

Export a visual for at least three real events: one improvement, one near-neutral change and one degradation, when the recorded data supports those categories. If a category is absent, label it absent instead of inventing it. Show the 2 parent trajectories and the child trajectory for each event. A child is not guaranteed to improve.

Acceptance criteria:
1. The synthetic example is bit-for-bit/angle-for-angle reproducible and transparent.
2. Real event logs replay to the same genomes and scores as their original evaluations.
3. Crossovers never change genotype length; every action remains within limits after mutation.
4. Logging can be disabled, and its overhead is limited/bounded.
5. Test initial generation, elites, random seeds, and event selection labels.
