# world_gen.py (high-level pseudo-code)
import numpy as np
from dataclasses import dataclass

@dataclass
class World:
    width: int
    height: int
    obstacle_mask: np.ndarray       # True = blocked
    # optional precomputations:
    # free_cells: list[(x,y)]
    # components: labels for connectivity


def make_map(cfg, rng, options=None) -> World:
    """
    Create the obstacle grid and any precomputed structures.
    Goal: random ~20% obstacles BUT still solvable / not too fragmented.
    """
    W, H = cfg.width, cfg.height
    target_density = cfg.obstacle_density  # 


    obstacles = place_obstacles(W, H, density=target_density, rng=rng)



        # 5) If good, return World
    world = World(width=W, height=H, obstacle_mask=obstacles)
        # optional: store free_cells list, CC labels, distance transforms, etc.
    return world


def place_obstacles(W, H, density, rng, *, k=8):
    """
    Option 1 (by construction):
    Grow one connected FREE region until reaching target free count.
    Returns obstacle_mask uint8: 0=free, 1=blocked.

    Anti-clutter heuristic:
    Choose frontier cells that have fewer already-free neighbors (less blobby).
    """

    free_target = int(round((1.0 - density) * H * W))
    free_target = max(1, min(H * W, free_target))

    obstacle = np.ones((H, W), dtype=np.uint8)      # 1=blocked
    in_frontier = np.zeros((H, W), dtype=bool)
    frontier = []

    nbrs = [(-1,0), (1,0), (0,-1), (0,1)]

    def add_frontier(y, x):
        if 0 <= y < H and 0 <= x < W and obstacle[y, x] == 1 and not in_frontier[y, x]:
            in_frontier[y, x] = True
            frontier.append((y, x))

    def carve(y, x):
        obstacle[y, x] = 0
        for dy, dx in nbrs:
            add_frontier(y + dy, x + dx)

    def free_neighbor_count(y, x):
        c = 0
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and obstacle[ny, nx] == 0:
                c += 1
        return c

    # start from a random seed
    sy, sx = int(rng.integers(0, H)), int(rng.integers(0, W))
    carve(sy, sx)
    free_count = 1

    carve(0, 0)
    while free_count < free_target and frontier:
        # sample k frontier cells and pick the least "blobby"
        idxs = rng.integers(0, len(frontier), size=min(k, len(frontier)))
        best_i = int(idxs[0])
        best_score = 1e9
        for i in idxs:
            y, x = frontier[int(i)]
            score = free_neighbor_count(y, x) + rng.random() * 1e-3  # tie-break
            if score < best_score:
                best_score = score
                best_i = int(i)

        # remove chosen frontier cell (swap-with-last)
        y, x = frontier[best_i]
        last = frontier.pop()
        if best_i < len(frontier):
            frontier[best_i] = last
        in_frontier[y, x] = False

        carve(y, x)
        free_count += 1


    return obstacle
