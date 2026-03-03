# core.py (pseudo-code)

import numpy as np
from .scenarios import make_map, spawn_agents
from .rewards import compute_rewards  # optional
from .render import render_world      # optional

class CoverageCore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = None

        # Persistent structures allocated once (optional optimization)
        self.world = None            # obstacle grid / continuous world
        self.coverage = None         # visited / coverage grid
        self.agent_state = {}        # agent -> state object
        self.t = 0

        # caches for step outputs
        self._rewards = {}
        self._terminations = {}
        self._truncations = {}
        self._infos = {}

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)

        # 1) Build scenario/map
        self.world = make_map(self.cfg, rng=self.rng, options=options)

        # 2) Initialize coverage memory (fog-of-war / visited)
        self.coverage = self._init_coverage(self.world)

        # 3) Spawn agents
        self.agent_state = spawn_agents(self.cfg, self.world, rng=self.rng)

        # 4) Reset time + caches
        self.t = 0
        self._clear_step_caches()

        # 5) Optionally compute initial coverage from spawn positions
        self._update_coverage_from_agents()

        # 6) Initialize infos for t=0
        self._infos = self._compute_infos()
    

    def step(self, actions: dict, alive_agents: list):
        self._clear_step_caches()

        # 1) Convert actions to intended moves/controls
        intended = self._decode_actions(actions, alive_agents)

        # 2) Apply physics / movement with collision handling
        #    (grid: move N/E/S/W; continuous: integrate velocity)
        self._apply_moves(intended, alive_agents)

        # 3) Update coverage grid based on sensors/FOV
        self._update_coverage_from_agents()

        # 4) Compute rewards (coverage gained, overlap penalty, etc.)
        self._rewards = compute_rewards(
            cfg=self.cfg,
            world=self.world,
            coverage=self.coverage,
            agent_state=self.agent_state,
            t=self.t,
            alive_agents=alive_agents,
        )

        # 5) Compute termination/truncation
        self._terminations = self._compute_terminations(alive_agents)
        self._truncations = self._compute_truncations(alive_agents)

        # 6) Compute infos (metrics) AFTER state updates
        self._infos = self._compute_infos()

        # 7) Advance time
        self.t += 1

    # ----- Observation / output getters -----
    def get_global_state(self):

        return
    def get_critic_state(self):
        # 1. Downsample the coverage map (100x100 -> 10x10)
        # This gives the Critic a "heat map" of where work is left
        reshaped = self.coverage.reshape(10, 10, 10, 10)
        zone_coverage = reshaped.mean(axis=(1, 3)) 
        
        # 2. Get Agent Zone indices
        drone_zone = (self.agent_state['drone'].pos // 10)
        car_zone = (self.agent_state['car'].pos // 10)
        
        # 3. Construct a feature vector for the Critic
        # You can flatten this for a standard MLP Critic 
        # or keep it as a graph for a GNN Critic.
        state = {
            "zone_map": zone_coverage.flatten(), # 100 values
            "agent_locs": np.concatenate([drone_zone, car_zone]), # 4 values
            "global_stats": [self.t / self.cfg.max_steps, self._coverage_percent()]
        }
        return np.concatenate(list(state.values()))

    def _build_observation(self, agent_id):
        s = self.agent_state[agent_id]
        
        # Define window size based on agent type
        win_size = 21 if "drone" in agent_id else 7
        
        # Extract views from the global "Truth" layers
        local_obs = self._extract_local_patch(self.world.obstacles, s.pos, win_size)
        local_cov = self._extract_local_patch(self.coverage, s.pos, win_size)
        
        # Stack them into a 'mini-image' for the Agent's CNN
        # Shape will be (2, win_size, win_size)
        visual_input = np.stack([local_obs, local_cov], axis=0)
        
        return {
            "image": visual_input,
            "coords": np.array([s.x / 100, s.y / 100], dtype=np.float32) # Normalized 0-1
        }
    
    def get_obs(self, agents):
        obs = {}
        for a in agents:
            obs[a] = self._build_observation(a)
        return obs

    def get_rewards(self, agents):
        return {a: float(self._rewards.get(a, 0.0)) for a in agents}

    def get_terminations(self, agents):
        return {a: bool(self._terminations.get(a, False)) for a in agents}

    def get_truncations(self, agents):
        return {a: bool(self._truncations.get(a, False)) for a in agents}

    def get_infos(self, agents):
        # You can include per-agent and shared metrics
        infos = {a: dict(self._infos.get(a, {})) for a in agents}
        return infos

    # ----- Key internal helpers -----
    def _extract_local_patch(self, layer, pos, window_size):
        """
        layer: 2D numpy array (e.g., self.world.obstacles)
        pos: (x, y) tuple of agent position
        window_size: int (7 for car, 21 for drone)
        """
        x, y = pos
        radius = window_size // 2
        
        # 1. Pad the layer with a constant (1.0 for obstacles/edges)
        # This creates a 'safety buffer' so we never go out of bounds
        padded_layer = np.pad(layer, pad_width=radius, mode='constant', constant_values=1.0)
        
        # 2. Adjust coordinates for the padding
        # Because we added 'radius' pixels to the top/left, (0,0) is now (radius, radius)
        nx, ny = x + radius, y + radius
        
        # 3. Slice the window
        patch = padded_layer[nx - radius : nx + radius + 1, 
                            ny - radius : ny + radius + 1]
        
        return patch

    def _build_observation(self, agent):
        # Typical coverage obs ideas:
        # - local patch of obstacle grid
        # - local patch of coverage grid (visited/fog)
        # - relative positions of nearby agents
        # - agent heading/battery/time remaining
        s = self.agent_state[agent]
        local_map = self._extract_local_patch(self.world.obstacles, s.pos)
        local_cov = self._extract_local_patch(self.coverage, s.pos)
        rel_agents = self._relative_agent_features(agent)
        return {
            "local_obstacles": local_map,
            "local_coverage": local_cov,
            "rel_agents": rel_agents,
            "self_state": [s.x, s.y, s.heading],
        }

    def _compute_terminations(self, alive_agents):
        # Termination usually means "task success" or "hard failure"
        done = {}
        if self._coverage_percent() >= self.cfg.target_coverage:
            for a in alive_agents:
                done[a] = True
        # optionally: agent-specific failure conditions
        return done

    def _compute_truncations(self, alive_agents):
        # Truncation typically means time limit reached
        trunc = {}
        if self.t >= self.cfg.max_steps:
            for a in alive_agents:
                trunc[a] = True
        return trunc

    def _compute_infos(self):
        # Put metrics here, not only reward terms
        # Example:
        global_cov = self._coverage_percent()
        infos = {"__common__": {"coverage": global_cov, "t": self.t}}
        for a, s in self.agent_state.items():
            infos[a] = {
                "collisions": s.collisions,
                "cells_covered_by_agent": s.covered_count,
            }
        return infos

    def render(self, mode="human"):
        return render_world(self.world, self.coverage, self.agent_state, mode=mode)

    def close(self):
        pass

    def _clear_step_caches(self):
        self._rewards = {}
        self._terminations = {}
        self._truncations = {}
        self._infos = {}

    # ... plus helpers: _decode_actions, _apply_moves,
    #     _update_coverage_from_agents, _coverage_percent, etc.