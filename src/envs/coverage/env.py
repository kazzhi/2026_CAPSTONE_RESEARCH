# env.py (pseudo-code)

from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ParallelEnv
from .core import CoverageCore
from .spaces import build_action_spaces, build_observation_spaces

#add action spaces for drone and car

def parallel_env(**kwargs):
    # 1) Build config with defaults
    cfg = EnvConfig(**kwargs)

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

        # PettingZoo requires stable agent names
        self.possible_agents = [f"agent_{i}" for i in range(cfg.num_agents)]

        # Core simulator does the real work
        self.core = CoverageCore(cfg)

        # Spaces must be dicts: {agent: space}
        self.action_spaces = build_action_spaces(cfg, self.possible_agents)
        self.observation_spaces = build_observation_spaces(cfg, self.possible_agents)

        # These are updated after reset/step
        self.agents = []

        from gymnasium import spaces

        # self.observation_spaces = {
        #     "drone": spaces.Dict({
        #         "observation": spaces.Box(low=0, high=1, shape=(2, 21, 21), dtype=np.float32),
        #         "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        #     }),
        #     "car": spaces.Dict({
        #         "observation": spaces.Box(low=0, high=1, shape=(2, 7, 7), dtype=np.float32),
        #         "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        #     })
        # }

    def observe(self, agent):
        return



    def reset(self, seed=None, options=None):
        # 1) Seed (PettingZoo expects determinism with seed)
        self.core.reset(seed=seed, options=options)

        # 2) Alive agents list is set on reset
        self.agents = self.possible_agents[:]

        # 3) Produce initial observations and infos
        obs = self.core.get_obs(self.agents)
        infos = self.core.get_infos(self.agents)
        return obs, infos

    def step(self, actions: dict):
        # actions: {agent_name: action}
        # (Optional) validate keys: must match current self.agents

        # 1) Advance simulation one timestep
        self.core.step(actions, alive_agents=self.agents)

        # 2) Pull results from core
        observations = self.core.get_obs(self.agents)
        rewards = self.core.get_rewards(self.agents)
        terminations = self.core.get_terminations(self.agents)
        truncations = self.core.get_truncations(self.agents)
        infos = self.core.get_infos(self.agents)

        # 3) Update self.agents: remove terminated/truncated agents
        self.agents = [
            a for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.core.render(mode=self.render_mode)

    def close(self):
        self.core.close()

    # Optional: required by some wrappers/tools
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]