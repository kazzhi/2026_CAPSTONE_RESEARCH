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
        
        pct = float(self.core._coverage_percent())
        count = int(self.core.covered_count)

        infos = {a: {"coverage_pct": pct, "covered_cells": count} for a in observations.keys()}
        infos["__common__"] = {"coverage_pct": pct, "covered_cells": count}


        return observations, infos

    def step(self, actions):
        obs, rew, term, trunc, infos = self.core.step(actions, alive_agents=self.agents)
        
        for new_agent in self.core._newly_spawned:
            if new_agent not in self.agents:
                self.agents.append(new_agent)

        self.agents = [a for a in self.agents if not (term.get(a, False) or trunc.get(a, False))]
        
        
        # formatted_infos = {}
        # #global_coverage = float(infos.get("__common__", {}).get("coverage", 0.0))
        # global_coverage = self.core._coverage_percent()
        # global_cov_count = int(self.core.covered_count)

        # for agent_id in obs.keys():
        #     # Start with the agent's specific info from core (collisions, etc.)
        #     agent_info = infos.get(agent_id, {}).copy()
            
        #     # Inject the global coverage metric for RLlib logging
        #     agent_info["rllib_coverage_metric"] = global_coverage
        #     agent_info["map_covered_count"] = global_cov_count
            
        #     formatted_infos[agent_id] = agent_info
        pct = float(self.core._coverage_percent())
        count = int(self.core.covered_count)

        # Keep existing per-agent infos if you had them, but only for alive agents
        fixed_infos = {}

        # Optional global info
        fixed_infos["__common__"] = {"coverage_pct": pct, "covered_cells": count}

        for a in obs.keys():
            d = infos.get(a, {})  # keep whatever core.step already placed
            d["coverage_pct"] = pct
            d["covered_cells"] = count
            fixed_infos[a] = d
        
        infos["__common__"]["t"] = self.core.t
        infos["__common__"]["alive_agents"] = list(self.agents)
        infos["__common__"]["drone_battery"] = {a: self.core.agent_state[a].battery for a in self.agents if "drone" in a}
        # fixed_infos["__common__"]["final_coverage_pct"] = pct
        # fixed_infos["__common__"]["final_covered_cells"] = count
        
        return obs, rew, term, trunc, fixed_infos
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    def render(self, mode='rgb_array'):
        return self.core.render_frame()
# Helper function for RLlib registration
def env_creator(config):
    # config here is the dictionary passed from RLlib
    cfg = CoverageConfig(**config)
    env = CoverageParallelEnv(cfg)
    return ParallelPettingZooEnv(env)
