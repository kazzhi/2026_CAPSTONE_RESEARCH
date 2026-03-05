from dataclasses import dataclass

@dataclass
class CoverageConfig:
    #Map specs
    max_map_attempts: int = 10
    width: int = 100
    height: int = 100
    obstacle_density: float = 0.195
    start_x: int = 0
    start_y: int = 0
    target_coverage: float = 95.0

    #Drone
    drone_fov: int = 21
    drone_battery_max: float = 100.0
    drone_move_cost: float = 1.0
    drone_idle_cost: float = 0.1
    drone_spawn_cost: float = 40.0

    #Car
    car_fov: int = 7
    car_move_cost: float = 0.005


    #RL REWARDS !!! NEEDS TWEAKINGG
    reward_per_cell: float = 1.0
    crash_penalty: float = -500.0
    step_penalty: float = -0.01
    target_coverage: float = 0.95
    max_steps: int = 5000
    battery_out_penalty: float = -500.0
    overlap_penalty: float = -1.0
    car_spawn_reward: float = 50.0


    

