# Task 1 — Refactor the existing core without changing behavior

Read `AGENTS.md`, `ARCHITECTURE.md`, and the relevant `buglab` modules. Upgrade the current architecture incrementally to support pluggable maps, pluggable fitness functions, traceable algorithm steps, and interactive playback in later tasks.

**Do not rewrite the simulator or add UI in this task.** Establish a reproducible baseline first, including a fixed-seed genome and its rollout. Refactor the minimal boundaries needed so geometry/physics, map definitions, fitness policy and experiment orchestration can evolve independently. Keep `Optimizer.ask()/tell()` scalar-only and preserve the existing CLI, GIF export and current default map behavior unless a compatibility wrapper is documented.

Constraints:
- Fixed action sequence, constant step length, relative turn angles in radians.
- Collision detection must prevent tunneling; a blocked move turns in place and consumes a time step.
- Don't couple optimizer internals to the map or scoring implementation.
- Preserve deterministic runs and stable result data structures where practical.
- Do not add heavyweight dependencies.

Acceptance criteria:
1. Existing test suite passes and new tests cover deterministic rollout, angle clamping, boundaries, and segment collisions.
2. A fixed-seed pre/post-refactor example has identical or documented intentionally corrected physics and scores.
3. `python -m buglab --generations 8 --population 16 --seeds 0 --methods ga random local` runs and produces expected existing artifacts.
4. Provide a concise diff summary and call out any behavior changes.
