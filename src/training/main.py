import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.envs.coverage.env import env_creator, CoverageParallelEnv, ParallelEnv, parallel_env
from src.envs.coverage.model import DroneCarHybridModel
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


# 1. Register the Model
ModelCatalog.register_custom_model("drone_car_cnn", DroneCarHybridModel)

# 2. Register the Env
def w_env_creator(config):
    # This calls your existing parallel_env function
    env = parallel_env(**config)
    return ParallelPettingZooEnv(env)

register_env("coverage_v0", w_env_creator)

# 3. Define the PPO Config
# We need the spaces to initialize the policies
# tmp_env = env_creator({})
# drone_obs = tmp_env.observation_space["drone_0"]
# drone_act = tmp_env.action_space["drone_0"]
# car_obs = tmp_env.observation_space["car_0"]
# car_act = tmp_env.action_space["car_0"]

raw_env = parallel_env(**{}) 

# Access the dictionaries directly instead of through the wrapper's .observation_space property
drone_obs = raw_env.observation_spaces["drone_0"]
drone_act = raw_env.action_spaces["drone_0"]
car_obs = raw_env.observation_spaces["car_0"]
car_act = raw_env.action_spaces["car_0"]

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    .environment("coverage_v0", env_config={"width": 100, "height": 100})
    .framework("torch")
    .multi_agent(
        policies={
            "drone_policy": (None, drone_obs, drone_act, {"model": {"custom_model": "drone_car_cnn"}}),
            "car_policy": (None, car_obs, car_act, {"model": {"custom_model": "drone_car_cnn"}}),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: 
            "drone_policy" if "drone" in agent_id else "car_policy",
    )
    .training(
        gamma=0.99,
        lr=1e-4,
        train_batch_size=4000,
        
    )
    .env_runners(num_env_runners=0) # Uses more CPUs for data collection
)

config.sgd_minibatch_size=12

# 4. Run Training
tune.run(
    "PPO",
    name="coverage_v1",
    config=config.to_dict(),
    stop={"training_iteration": 500},
    checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=10)
)