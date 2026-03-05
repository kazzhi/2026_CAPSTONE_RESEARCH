import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
import os

from src.envs.coverage.env import env_creator, CoverageParallelEnv, ParallelEnv, parallel_env
from src.envs.coverage.model import DroneCarHybridModel
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from .callbacks import CoverageCallbacks

class CoverageCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # Initialize the list so 'on_episode_step' doesn't throw a KeyError
        episode.user_data["frames"] = []
        episode.user_data["coverage_pct"] = 0.0
        episode.user_data["covered_cells"] = 0
        
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        agents = episode.get_agents()
        if not agents:
            return

        # agents can be ["drone_0", ...] or [("drone_0", ...)] depending on RLlib internals
        a0 = agents[0]
        agent_id = a0 if isinstance(a0, str) else a0[0]

        info = episode.last_info_for(agent_id)
        if not info:
            return

        if "coverage_pct" in info:
            episode.user_data["coverage_pct"] = info["coverage_pct"]
        if "covered_cells" in info:
            episode.user_data["covered_cells"] = info["covered_cells"]
        
        raw_env = base_env.get_sub_environments()[0]
        
        # Only record frames for the first worker to save memory/IO
        if episode.length %100 == 0:
            if worker.worker_index <= 1:
                frame = raw_env.render()
                episode.user_data["frames"].append(frame)

        

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        pct = float(episode.user_data.get("coverage_pct", 0.0))
        count = int(episode.user_data.get("covered_cells", 0))
        print(f"--- EPISODE ENDED | FINAL COVERAGE: {pct:.2f}% and {count} cells ---")

        # Optional: log to TensorBoard as custom metrics
        episode.custom_metrics["final_coverage_pct"] = pct
        episode.custom_metrics["final_covered_cells"] = count
        
        if worker.worker_index <= 1 and "frames" in episode.user_data:
            frames = episode.user_data["frames"]
            if not frames:
                return

            # Save the final state as a PNG using Matplotlib
            # This avoids the need for Pillow
            plt.figure(figsize=(8, 8))
            plt.imshow(frames[-1])
            plt.axis('off')
            plt.title(f"End of Episode {episode.episode_id}")
            
            os.makedirs("render_outputs", exist_ok=True)
            plt.savefig(f"render_outputs/episode_{episode.episode_id}_final.png")
            plt.close()

            print(f"Saved render for episode {episode.episode_id}")
        
        
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
    .environment("coverage_v0", env_config={"width": 30, "height": 30})
    .callbacks(CoverageCallbacks)
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
        gamma=0.999,
        lr=1e-4,
        train_batch_size=4000,
        entropy_coeff=0.05, 
        # Optional: Gradually reduce entropy as they learn
        # entropy_coeff_schedule=[[0, 0.05], [200000, 0.001]],
        kl_coeff=0.2,
        clip_param=0.3,
        num_sgd_iter=10,
        
    )
    .env_runners(num_env_runners=5) # Uses more CPUs for data collection
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