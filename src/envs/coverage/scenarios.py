# world_gen.py (high-level pseudo-code)

from dataclasses import dataclass

@dataclass
class World:
    width: int
    height: int
    obstacles: "bool[H][W]"         # True = blocked
    # optional precomputations:
    # free_cells: list[(x,y)]
    # components: labels for connectivity


def make_map(cfg, rng, options=None) -> World:
    """
    Create the obstacle grid and any precomputed structures.
    Goal: random ~20% obstacles BUT still solvable / not too fragmented.
    """
    W, H = cfg.width, cfg.height
    target_density = cfg.obstacle_density  # 0.20

    for attempt in range(cfg.max_map_attempts):
        # 1) Start empty grid
        obstacles = [[False for x in range(W)] for y in range(H)]

        # 2) Place obstacles
        # Option A: independent Bernoulli per cell
        # Option B: "blobby" obstacles by dropping seeds and expanding
        obstacles = place_obstacles_random(obstacles, density=target_density, rng=rng)

        # 3) Enforce constraints:
        # - keep border free (optional)
        # - ensure not too many isolated single free cells
        # - ensure free space connectivity >= some threshold
        if cfg.keep_border_free:
            clear_border(obstacles)

        # 4) Validate map quality
        free_ratio = compute_free_ratio(obstacles)
        if not is_density_close_enough(free_ratio, 1 - target_density, tol=cfg.density_tol):
            continue

        # Connectivity check (recommended for coverage)
        if cfg.require_connected_free_space:
            largest_cc = largest_connected_component_size(obstacles)
            if largest_cc < cfg.min_connected_free_cells:
                continue

        # 5) If good, return World
        world = World(width=W, height=H, obstacles=obstacles)
        # optional: store free_cells list, CC labels, distance transforms, etc.
        return world

    # If all attempts fail, fall back (e.g., empty obstacles or relaxed constraint)
    return World(width=W, height=H, obstacles=obstacles_fallback(W, H, rng, target_density))