from dataclasses import dataclass


@dataclass(frozen=True)
class MpcAction:
    
    acceleration: float  # [m/s²]
    steering_angle: float  # [rad]
