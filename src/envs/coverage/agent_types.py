
from dataclasses import dataclass

@dataclass
class AgentState:
    type: str
    x: int
    y: int
    pos: tuple
    battery: float
    is_active: bool 
    is_moving: bool 
    collisions: int 
    num_spawns: int
    # ... other stats