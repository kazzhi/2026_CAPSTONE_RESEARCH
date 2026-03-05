# main.py - Revised Best-Only Logic
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import os
import matplotlib.pyplot as plt


# main.py

class CoverageCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.best_frame = None
        self.max_cov = -1.0

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        # Only record frames for worker 1 to save memory
        if worker.worker_index <= 1 and episode.length % 100 == 0:
            raw_env = base_env.get_sub_environments()[0]
            # Store the frame in user_data temporarily
            episode.user_data["last_frame"] = raw_env.render()

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        # Check if this episode outperformed others in this iteration
        cov = episode.user_data.get("coverage_pct", 0)
        if cov > self.max_cov:
            self.max_cov = cov
            self.best_frame = episode.user_data.get("last_frame")

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result["training_iteration"]
        if self.best_frame is not None:
            import matplotlib.image as mpimg
            os.makedirs("render_outputs1/best_per_iter", exist_ok=True)
            path = f"render_outputs1/best_per_iter/iter_{iteration:03d}_cov_{self.max_cov:.1f}.png"
            mpimg.imsave(path, self.best_frame)
            
        # Reset for the next iteration
        self.max_cov = -1.0
        self.best_frame = None