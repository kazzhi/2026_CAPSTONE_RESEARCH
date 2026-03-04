# env.py (pseudo-code)

from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ParallelEnv
from .core import CoverageCore
from .spaces import get_action_space, get_observation_space
from .render import render
from .config import CoverageConfig

#add action spaces for drone and car

def parallel_env(**kwargs):
    # 1) Build config with defaults
    cfg = CoverageConfig(**kwargs)

    # 2) Construct the actual PettingZoo env wrapper object
    env = CoverageParallelEnv(cfg)

    # 3) Apply standard wrappers (optional but recommended)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)  # mostly for AEC, but ok to keep
    return env


class CoverageParallelEnv(ParallelEnv):
    metadata = {
        "name": "coverage_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }
    from gymnasium import spaces


    def __init__(self, cfg):

        self.cfg = cfg

        # FIX 1: Use the EXACT names defined in core.py
        # Match your core.py: ["drone_0", "drone_1", "car_0", "car_1", "car_2", "car_3"]
        self.possible_agents = ["drone_0", "drone_1", "car_0", "car_1", "car_2", "car_3"]

        self.core = CoverageCore(cfg)

        # FIX 2: Pass individual agent names to your spaces functions
        self.action_spaces = {
            a: get_action_space(a) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: get_observation_space(a, self.cfg) for a in self.possible_agents
        }

        self.agents = []
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        # 1) Seed (PettingZoo expects determinism with seed)
        self.core.reset(seed=seed, options=options)

        # 2) Alive agents list is set on reset
        self.agents = [a for a in self.possible_agents if "drone" in a]

        # 3) Produce initial observations and infos
        observations = {a: self.core._build_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
            
        return observations, infos

    def step(self, actions: dict):
        # actions: {agent_name: action}
        # (Optional) validate keys: must match current self.agents

        # 1) Advance simulation one timestep
        observations, rewards, terminations, truncations, infos = self.core.step(actions, alive_agents=self.agents)

        # 2) Pull results from core
        for new_agent in self.core._newly_spawned:
            if new_agent not in self.agents:
                self.agents.append(new_agent)

        # 3) Update self.agents: remove terminated/truncated agents
        self.agents = [
            a for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):

        return self.core.render_frame(mode=self.render_mode)

    def close(self):
        self.core.close()

    # Optional: required by some wrappers/tools
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]