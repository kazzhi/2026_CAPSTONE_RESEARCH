
import numpy as np
from .scenarios import make_map
from .rewards import compute_rewards  # optional
# from .render import render_world      # optional
from .agent_types import AgentState
# from gymnasium import spaces
# from .render import MatplotlibGridRenderer, RenderConfig


class CoverageCore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = None

        # Persistent structures allocated once (optional optimization)
        self.world = None            # obstacle grid / continuous world
        self.coverage = None         # visited / coverage grid
        self.agent_state = {}        # agent -> state object
        self.t = 0

        self.total_reachable = 0
        self.covered_count = 0
        self.possible_agents = []

        # caches for step outputs
        self._rewards = {}
        self._terminations = {}
        self._truncations = {}
        self._infos = {}
        self._newly_spawned = []

        
        self._renderer = MatplotlibGridRenderer(
            height=self.cfg.height,
            width=self.cfg.width,
            cfg=RenderConfig(cell_px=8)
        )

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)

        # 1) Build scenario/map
        self.world = make_map(self.cfg, rng=self.rng, options=options)
        self.coverage = self._init_coverage(self.world)

        self.total_reachable = np.sum(self.world.obstacle_mask == 0)
        self.covered_count = 0

        self.possible_agents = ["drone_0","drone_1","car_0","car_1","car_2","car_3"]
        self.agent_state = {}

        for a_id in self.possible_agents:
            if "drone" in a_id:
                self.agent_state[a_id] = AgentState(type = "drone", x=self.cfg.start_x, y=self.cfg.start_y, battery=100.0, is_active=True)
            else:
                # Cars start inactive and "off-map"
                self.agent_state[a_id] = AgentState(type = "car", x=-1, y=-1, battery=0.0, is_active=False)

        # 4) Reset time + caches
        self.t = 0
        self._clear_step_caches()

        # 5) Optionally compute initial coverage from spawn positions
        self._update_coverage_from_agents()

        # 6) Initialize infos for t=0
        self._infos = self._compute_infos()

        return 
    

    def step(self, actions: dict, alive_agents: list):
        self.t += 1
        self._clear_step_caches()

        # 1) Convert actions to intended moves/controls
        intended = self._decode_actions(actions)
        self._apply_moves(intended, alive_agents)

        new_cells, overlap_cells = self._update_coverage_from_agents()

        # 4) Compute rewards (coverage gained, overlap penalty, etc.)
        self._rewards = compute_rewards(
            cfg=self.cfg,
            world=self.world,
            coverage=self.coverage,
            agent_state=self.agent_state,
            t=self.t,
            alive_agents=alive_agents,
            new_cells = new_cells,
            overlap_cells = overlap_cells
        )

        # 5) Compute termination/truncation
        self._terminations = self._compute_terminations(alive_agents)
        self._truncations = {a: self.t >= self.cfg.max_steps for a in alive_agents}

        # 6) Compute infos (metrics) AFTER state updates
        obs = {a: self._build_observation(a) for a in alive_agents}

    
        return obs, self._rewards, self._terminations, self._truncations, self._infos
    

    # ----- Observation / output getters -----
    def get_global_state(self):

        return


    def _build_observation(self, agent_id):
        s = self.agent_state[agent_id]
        cfg = self.cfg
        
        # Determine window size
        win_size = cfg.drone_fov if "drone" in agent_id else cfg.car_fov
        
        # 1. Visual observations (the crops we made earlier)
        local_map = self._extract_local_patch(self.world.obstacle_mask, s.pos, win_size)
        local_cov = self._extract_local_patch(self.coverage, s.pos, win_size)
        image_data = np.stack([local_map, local_cov], axis=0).astype(np.float32)
        
        # 2. Vector observations (the "Internal" state)
        # Battery and Pos are normalized 0 to 1 for better NN convergence
        vector_data = np.array([
            s.battery / 100.0,
            s.x / cfg.width,
            s.y / cfg.height
        ], dtype=np.float32)
        
        return {
            "image": image_data,
            "vector": vector_data
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
        patch = padded_layer[ny-radius:ny+radius+1, nx-radius:nx+radius+1]
        
        return patch

    def _compute_terminations(self, alive_agents):
        # Termination usually means "task success" or "hard failure"
        terminations = {}
        for a in alive_agents:
                s = self.agent_state[a]
                # Agent is done if they crashed, ran out of battery, or map is 100%
                if not s.is_active or self._coverage_percent() >= self.cfg.target_coverage:
                    terminations[a] = True
                else:
                    terminations[a] = False
        return terminations

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

    def close(self):
        pass

    def _clear_step_caches(self):
        self._rewards = {}
        self._terminations = {}
        self._truncations = {}
        self._infos = {}
        self._newly_spawned = []
    
    def _apply_moves(self, actions, alive_agents):
        move_map = {0: (0,0), 1: (-1,0), 2: (1,0), 3: (0,-1), 4: (0,1)}

        for agent_id in alive_agents:
            s = self.agent_state[agent_id]
            if not s.is_active:
                continue
                
            action = actions.get(agent_id, 0)
            
            # --- 1. HANDLE SPAWNING (Drone Only) ---
            if "drone" in agent_id and action == 5:
                available_car = next((c_id for c_id in self.agent_state 
                                    if "car" in c_id and not self.agent_state[c_id].is_active 
                                    and self.agent_state[c_id].battery == 0.0), None)
                
                can_spawn = (s.battery >= self.cfg.drone_spawn_cost and 
                            self.world.obstacle_mask[s.y, s.x] == 0)
                
                if available_car and can_spawn:
                    s.battery -= self.cfg.drone_spawn_cost
                    target_car = self.agent_state[available_car]
                    target_car.x, target_car.y = s.x, s.y
                    target_car.pos = (s.x, s.y)
                    target_car.battery = 100.0
                    target_car.is_active = True
                    self._newly_spawned.append(available_car)
                
                # Drone spent its turn spawning; skip movement
                self._handle_battery_and_status(s, is_moving=False)
                continue 

            # --- 2. HANDLE MOVEMENT (Car & Drone) ---
            dx, dy = move_map.get(action, (0,0))
            new_x = np.clip(s.x + dx, 0, self.cfg.width - 1)
            new_y = np.clip(s.y + dy, 0, self.cfg.height - 1)

            if "car" in agent_id:
                if self.world.obstacle_mask[new_y, new_x] == 1:
                    # HIT OBSTACLE: Terminate immediately
                    s.is_active = False
                    s.collisions += 1
                else:
                    s.x, s.y = new_x, new_y
            else:
                # Drone ignores obstacles
                s.x, s.y = new_x, new_y

            # --- 3. UPDATE STATUS ---
            is_moving = (dx != 0 or dy != 0)
            s.is_moving = is_moving
            self._handle_battery_and_status(s, is_moving, agent_id)
            s.pos = (s.x, s.y)

    def _handle_battery_and_status(self, s, is_moving, agent_id):
        # Apply costs from your config
        if "drone" in agent_id :
            s.battery -= self.cfg.drone_move_cost if is_moving else self.cfg.drone_idle_cost
        else:
            s.battery -= self.cfg.car_move_cost if is_moving else 0

        # Terminate if empty
        if s.battery <= 0:
            s.battery = 0
            s.is_active = False
    
    def _init_coverage(self, world):
        """Creates a zero-filled grid matching the map size."""
        # 0 = Unvisited, 1 = Visited
        return np.zeros((self.cfg.width, self.cfg.height), dtype=np.float32)

    def _decode_actions(self, actions):
        """
        Converts raw RL actions (0,1,2...) into a standardized format.
        Useful if you ever want to change the action mapping in one place.
        """
        return {a_id: int(act) for a_id, act in actions.items()}

    def _update_coverage_from_agents(self):
        """Updates the coverage grid based on where active agents are standing."""
        newly_covered_this_step = 0
        overlap_cells = 0
        for a_id, s in self.agent_state.items():
            if s.is_active:
                
                if self.world.obstacle_mask[s.y, s.x] == 0 and self.coverage[s.y, s.x] == 0:
                    self.coverage[s.y, s.x] = 1.0
                    self.covered_count += 1
                    newly_covered_this_step += 1
                elif self.world.obstacle_mask[s.y, s.x] == 0 and self.coverage[s.y, s.x] == 1:
                    overlap_cells += 1
        return (newly_covered_this_step, overlap_cells)

    def _coverage_percent(self):
        """Calculates what percentage of the reachable map is covered."""
        # Total cells minus obstacles
        if self.total_reachable == 0: return 0.0
        return self.covered_count / self.total_reachable
    
    def render_frame(self, mode):

        step_reward = float(sum(self._rewards.values())) if hasattr(self, "_rewards") else None

        infos = {"__common__": {
            "t": getattr(self, "t", 0),
            "coverage": float(self.coverage.mean()) if hasattr(self, "coverage") else 0.0,
        }}

        return self._renderer.render_frame(
            obstacle_mask=self.world.obstacle_mask,
            coverage=self.coverage,
            agent_state=self.agent_state,
            step_reward=step_reward,
            infos=infos,
            drone_fov=self.cfg.drone_fov,  # 21
            car_fov=self.cfg.car_fov       # 7
        )
    