from dataclasses import dataclass

@dataclass
class CoverageConfig:
    #Map specs
    max_map_attempts: int = 10
    width: int = 30
    height: int = 30
    obstacle_density: float = 0.10
    start_x: int = 0
    start_y: int = 0
    target_coverage: float = 95.0

    #Drone
    drone_fov: int = 11
    drone_battery_max: float = 100.0
    drone_move_cost: float = 0.5
    drone_idle_cost: float = 0.05
    drone_spawn_cost: float = 35.0

    #Car
    car_fov: int = 7
    car_move_cost: float = 0.001


    #RL REWARDS !!! NEEDS TWEAKINGG
    reward_per_cell: float = 10.0
    crash_penalty: float = -500.0
    step_penalty: float = -0.01
    max_steps: int = 5000
    battery_out_penalty: float = -500.0
    overlap_penalty: float = -0.005
    car_spawn_reward: float = 500.0


    

