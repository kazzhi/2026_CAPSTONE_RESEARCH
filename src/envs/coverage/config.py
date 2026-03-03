from dataclasses import dataclass

@dataclass
class CoverageConfig:
    width: int = 100
    height: int = 100
    obstacle_density: float = 19.5

    #Drone
    drone_fov: int = 21
    drone_battery_max: float = 100.0
    drone_move_cost: float = 1.0
    drone_idle_cost: float = 0.1
    drone_spawn_cost: float = 40.0

    #Car
    car_fov: int = 7
    car_move_cost: float = 0.0

    #RL REWARDS !!! NEEDS TWEAKINGG
    reward_per_cell: float = 1.0
    crash_penalty: float = -500.0
    step_penalty: float = -0.01
    target_coverage: float = 0.95
    max_steps: int = 1000

    

