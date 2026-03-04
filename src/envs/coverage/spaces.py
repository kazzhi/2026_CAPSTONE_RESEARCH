from gym import spaces
import numpy as np



def get_observation_space(agent_id, cfg):
    """Returns the Dict space for a specific agent."""
    # Drone: 21x21, Car: 7x7
    win_size = cfg.drone_fov if "drone" in agent_id else cfg.car_fov
    
    return spaces.Dict({
        # 2 Channels: [Obstacles, Coverage]
        "image": spaces.Box(low=0, high=1, shape=(2, win_size, win_size), dtype=np.float32),
        # Vector: [Battery, Normalized_X, Normalized_Y]
        "vector": spaces.Box(low=1, high=1, shape=(4,), dtype=np.float32)
    })

def get_action_space(agent_id):
    """Returns the Discrete space for a specific agent."""
    if "drone" in agent_id:
        # 0:Stay, 1:N, 2:S, 3:E, 4:W, 5:SPAWN
        return spaces.Discrete(6)
    else:
        # Cars cannot spawn anything
        return spaces.Discrete(5)

