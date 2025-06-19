import numpy as np
from dataclasses import dataclass

@dataclass
class AlgorithmConfig:
    NX: int = 4  # State dimension
    NU: int = 2  # Actions dimension
    T: int = 5  # Horizon length
    R: np.ndarray = np.diag([0.01, 0.01])  # Input cost matrix
    Rd: np.ndarray = np.diag([0.01, 1.0])  # Input difference cost matrix
    Q: np.ndarray = np.diag([1.0, 1.0, 0.5, 0.5])  # State cost matrix
    Qf: np.ndarray = Q  # State final matrix
    GOAL_DIS: float = 1.5  # Goal distance to consider 'reached' [m]
    STOP_SPEED: float = 0.5 / 3.6  # [m/s]
    MAX_TIME: float = 500.0  # Maximum simulation time [s]
    MAX_ITER: int = 3  # Maximum MPC loop iterations
    DU_TH: float = 0.1  # Iteration finish param
    TARGET_SPEED: float = 10.0 / 3.6  # [m/s]
    N_IND_SEARCH: int = 10  # Search index number
    DT: float = 0.2  # Time diff [s]

@dataclass
class VehicleConfig:
    LENGTH: float = 4.5  # [m]
    WIDTH: float = 2.0  # [m]
    BACKTOWHEEL: float = 1.0  # Distance from back to rear wheel axis [m]
    WHEEL_LEN: float = 0.3  # [m]
    WHEEL_WIDTH: float = 0.2  # [m]
    TREAD: float = 0.7  # Distance between same-axis wheels [m]
    WB: float = 2.5  # Distance between axi [m]
    MAX_STEER: float = np.deg2rad(45.0)  # [rad]
    MAX_DSTEER: float = np.deg2rad(30.0)  # [rad/s]
    MAX_SPEED: float = 55.0 / 3.6  # [m/s]
    MIN_SPEED: float = -20.0 / 3.6  # [m/s]
    MAX_ACCEL: float = 1.0  # [m/s²]
    
@dataclass
class SimulationConfig:
    SHOW_ANIMATION: bool = True  # Show animation

default_algorithm_config = AlgorithmConfig()
default_vehicle_config = VehicleConfig()
default_simulation_config = SimulationConfig()
