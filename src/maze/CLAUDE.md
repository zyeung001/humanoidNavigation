# src/maze/ — Procedural Maze Generation + Navigation

## Pipeline

```
maze_generator.py / maze_maps.py   →   maze_mjcf.py   →   solver.py   →   navigation_controller.py
(generate grid)                        (grid → MJCF XML)   (A* path)      (pure pursuit waypoints)
```

## Files

### `maze_generator.py` — Procedural Generation

Two algorithms:
- `generate_maze_dfs(rows, cols)` — Depth-first search (long corridors, fewer branches)
- `generate_maze_prims(rows, cols)` — Prim's algorithm (more branching, organic feel)

Helper constructors: `open_arena(rows, cols)`, `corridor(length)`.

Grid format: 2D numpy array where `1` = wall, `0` = open. Start at `(1,1)`, goal at bottom-right open cell.

### `maze_maps.py` — Predefined Layouts

Hand-designed grids: `CORRIDOR`, `L_MAZE`, `U_MAZE`, `OPEN`, `MEDIUM_MAZE`.

### `maze_mjcf.py` — MuJoCo XML Generation

`MazeMJCFGenerator` converts a grid into a full MJCF XML string with:
- Merged wall segments (adjacent walls consolidated into single geoms for efficiency)
- Configurable cell size, wall height, wall thickness
- Floor plane and lighting

### `solver.py` — A* Pathfinding

`solve(grid, start, goal)` returns a list of `(row, col)` waypoints. Converts grid coordinates to world positions via cell size.

### `navigation_controller.py` — Pure Pursuit

`NavigationController` steers the frozen walking policy through waypoints:
- Lookahead-based waypoint selection
- Heading error → yaw rate command
- Stop-turn-walk strategy: stops walking when heading error > 25deg, turns in place, resumes when < 12deg
- Outputs `(vx, vy, yaw_rate)` for the walking policy

### `maze_renderer.py` — Top-Down Visualization

`MazeRenderer` draws a minimap overlay showing the maze grid, agent position, waypoints, and planned path.
