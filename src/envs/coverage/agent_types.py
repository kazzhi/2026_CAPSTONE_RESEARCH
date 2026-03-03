
from dataclasses import dataclass

@dataclass
class AgentState:
    x: int
    y: int
    pos: tuple
    battery: float
    is_active: bool = True
    collisions: int = 0
    # ... other stats