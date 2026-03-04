from src.envs.coverage.env import parallel_env
from src.envs.coverage.config import CoverageConfig

from marllib import marl


# 1. Register your custom environment
marl.register_env("coverage_mission", parallel_env)

# 2. Initialize the algorithm (IPPO is highly recommended for decentralized agents)
# This will utilize your GPU for gradient updates
algo = marl.algos.ippo(hyper_params={
    "batch_size": 4000,
    "lr": 5e-5,
})

# 3. Build the environment with your config
env = marl.make_env(
    environment_name="coverage_mission",
    map_name="default_100x100",
    force_coop=True,
    env_args={
        "width": 100,
        "height": 100,
        "obstacle_density": 0.195,
    }
)

# 4. Map policies: 'drone' agents get drone_net, 'car' agents get car_net
policy_mapping = {
    "all_scenario": {
        "description": "Heterogeneous Drone-Car Setup",
        "team_prefix": ("drone", "car"),
        "mapping_rules": {
            "drone": "drone_policy",
            "car": "car_policy",
        }
    }
}

# 5. Launch training
algo.fit(env, model="cnn", stop={"training_iteration": 500}, share_policy="group")