from mpc_config import AlgorithmConfig, VehicleConfig, default_algorithm_config, default_vehicle_config
import numpy as np
import math

from state import State

class MPCController:    
    def __init__(
        self,
        mpc_config: AlgorithmConfig = default_algorithm_config,
        vehicle_config: VehicleConfig = default_vehicle_config):
        
        self._mpc_config = mpc_config
        self._vehicle_config = vehicle_config


    def get_linear_model_matrix(self, v: float, phi: float, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = self._mpc_config.DT
        wb = self._vehicle_config.WB
        nx = self._mpc_config.NX
        nu = self._mpc_config.NU

        a = np.zeros((nx, nx))
        a[0, 0] = 1.0
        a[1, 1] = 1.0
        a[2, 2] = 1.0
        a[3, 3] = 1.0
        a[0, 2] = dt * math.cos(phi)
        a[0, 3] = -dt * v * math.sin(phi)
        a[1, 2] = dt * math.sin(phi)
        a[1, 3] = dt * v * math.cos(phi)
        a[3, 2] = dt * math.tan(delta) / wb

        b = np.zeros((nx, nu))
        b[2, 0] = dt
        b[3, 1] = dt * v / (wb * math.cos(delta) ** 2)

        c = np.zeros(nx)
        c[0] = dt * v * math.sin(phi) * phi
        c[1] = -dt * v * math.cos(phi) * phi
        c[3] = -dt * v * delta / (wb * math.cos(delta) ** 2)

        return a, b, c
    
    def update_state(self, state: State, a, delta):

        min_speed = self._vehicle_config.MIN_SPEED
        max_speed = self._vehicle_config.MAX_SPEED
        max_steer = self._vehicle_config.MAX_STEER
        wb = self._vehicle_config.WB
        dt = self._mpc_config.DT
        
        # input check
        if delta >= self._vehicle_config.MAX_STEER:
            delta = max_steer
        elif delta <= -max_steer:
            delta = -max_steer

        state.x = state.x + state.v * math.cos(state.yaw) * dt
        state.y = state.y + state.v * math.sin(state.yaw) * dt
        state.yaw = state.yaw + state.v / wb * math.tan(delta) * dt
        state.v = state.v + a * dt

        if state.v > max_speed:
            state.v = max_speed
        elif state.v < min_speed:
            state.v = min_speed

        return state
