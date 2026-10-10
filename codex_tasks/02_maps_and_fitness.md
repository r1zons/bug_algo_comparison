# Task 2 — Maps, reward shaping and independent outcome metrics

Add named map presets `open_field`, `two_bridges`, `spiral`, `narrow_gate`, and `dead_end`, without changing optimizer APIs. The same dynamics and collision rules must apply on every map. Keep the original map available as `default`.

Design meaningful and **reachable** layouts under the actual constraints (initial heading, max turn, fixed step length, episode horizon, beetle radius), not just geometrically open corridors. For the spiral, distinguish the visually short but blocked Euclidean direction from the valid winding corridor. Add a map gallery generator for the talk.

Make fitness strategies explicit and configurable. Include at least:
- `euclidean`: final or best Euclidean proximity to the goal.
- `success_distance`: success-priority reward, time-to-goal for successes and proximity for failures.
- `collision_penalty`: configurable obstacle penalty combined with a documented base reward.
- `checkpoints`: ordered progress through optional map-specific checkpoints.
- `geodesic`: obstacle-aware distance-field shaping (existing implementation, appropriately documented).
- `smoothness`: configurable turn-change penalty composed with a base reward.

Keep the physical goal test **independent of reward**. Use consistent shared metrics including goal_reached, steps_to_goal, collision_count, actual_path_length, final_distance, and number of fitness evaluations. Do not directly compare scalar scores from different reward strategies. Document where checkpoints and geodesic shaping supply extra domain knowledge and why they can change the result.

Acceptance criteria:
1. All six maps (including default) render with start, goal and obstacles.
2. Automated tests verify collision prevention, map validity and representative reachable trajectories, including spiral/two-bridges.
3. Tests validate each reward variant, including that reaching the goal takes priority in `success_distance`.
4. The CLI can choose map and fitness by name; configs/seeds are recorded.
5. Demo: same algorithm and map with Euclidean versus geodesic/checkpoint shaping, evaluated with the **same outcome metrics**. Do not promise that any one wins on every seed.
