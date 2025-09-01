from dataclasses import dataclass
from geometry import State

@dataclass(frozen=True)
class MpcAction:
    
    acceleration: float  # [m/s²]
    steering_angle: float  # [rad]


class MovementHistory:

    def __init__(self, initial_state: State):
        self._states = [initial_state]
        self._last_prediction: list[MpcAction] = []
        self._last_target_index = 0
        self._reached_goal = False
        self._times: list[float] = []
        self._ticked_time = 0.0

    @property
    def ticked_time(self) -> float:
        return self._ticked_time

    @property
    def recorded_times(self) -> list[float]:
        return self._times

    @property
    def last_target_index(self):
        return self._last_target_index
    
    @last_target_index.setter
    def last_target_index(self, value: int):
        self._last_target_index = value

    @property
    def last_prediction(self) -> list[MpcAction]:
        return self._last_prediction

    @last_prediction.setter
    def last_prediction(self, value: list[MpcAction]):
        self._last_prediction = value

    @property
    def reached_goal(self) -> bool:
        return self._reached_goal

    @reached_goal.setter
    def reached_goal(self, value: bool):
        self._reached_goal = value

    def add_state(self, state: State, time_tick: float):
        self._states.append(state)
        self._ticked_time += time_tick
        self._times.append(self.ticked_time)

    def get_state_history(self) -> list[State]:
        return self._states
