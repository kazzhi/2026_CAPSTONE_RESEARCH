# env.py
from pettingzoo.utils.env import ParallelEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from .core import CoverageCore
from .spaces import get_action_space, get_observation_space
from .config import CoverageConfig

# src/envs/coverage/env.py
def parallel_env(**kwargs):
    cfg = CoverageConfig(**kwargs)
    env = CoverageParallelEnv(cfg)
    return env

class CoverageParallelEnv(ParallelEnv):
    metadata = {"name": "coverage_v0", "render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, cfg):
        self.cfg = cfg
        self.possible_agents = ["drone_0", "drone_1", "car_0", "car_1", "car_2", "car_3"]
        self.core = CoverageCore(cfg)

        # RLlib needs these as dict attributes
        self.action_spaces = {a: get_action_space(a) for a in self.possible_agents}
        self.observation_spaces = {a: get_observation_space(a, self.cfg) for a in self.possible_agents}
        
        self.agents = []

    def reset(self, seed=None, options=None):
        self.core.reset(seed=seed, options=options)
        self.agents = [a for a in self.possible_agents if "drone" in a]
        observations = {a: self.core._build_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        obs, rew, term, trunc, infos = self.core.step(actions, alive_agents=self.agents)
        
        for new_agent in self.core._newly_spawned:
            if new_agent not in self.agents:
                self.agents.append(new_agent)

        self.agents = [a for a in self.agents if not (term.get(a, False) or trunc.get(a, False))]
        return obs, rew, term, trunc, infos
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

# Helper function for RLlib registration
def env_creator(config):
    # config here is the dictionary passed from RLlib
    cfg = CoverageConfig(**config)
    env = CoverageParallelEnv(cfg)
    return ParallelPettingZooEnv(env)