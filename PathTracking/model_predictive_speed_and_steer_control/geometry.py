import math
import numpy as np
from dataclasses import dataclass

from utils.angle import angle_mod

def pi_2_pi(angle):
    return angle_mod(angle)


def get_nparray_from_matrix(x: list[float]) -> np.typing.NDArray[np.float64]:
    return np.array(x).flatten()

@dataclass
class Pos2D:

    x: float
    y: float

@dataclass
class SplinePoint(Pos2D):

    yaw: float
    curvature: float

@dataclass(frozen=False)
class State(Pos2D):
    """
    vehicle state class
    """
    yaw: float = 0.0
    v: float = 0.0
    predelta = None
    
    @property
    def components(self):
        return [self.x, self.y, self.v, self.yaw]

@dataclass
class Route:
    """
    Route class to hold spline points
    """
    
    spline_points: list[SplinePoint]
    _cx: list[float]
    _cy: list[float]
    _cyaw: list[float]
    _ck: list[float]

    def __init__(self, spline_points: list[SplinePoint] = []):
        self.spline_points = []
        self._cx, self._cy, self._cyaw, self._ck = [], [], [], []
        for spline_point in spline_points:
            self.add_point(spline_point)

    def get_spline_data(self) -> tuple[list[float], list[float], list[float], list[float]]:
        return self._cx, self._cy, self._cyaw, self._ck
    
    def add_point(self, point: SplinePoint):
        self.spline_points.append(point)
        self._cx.append(point.x)
        self._cy.append(point.y)
        self._cyaw.append(point.yaw)
        self._ck.append(point.curvature)
    
    def get_smooth_path(self) -> 'Route':
        
        if (len(self.spline_points) == 0):
            return Route()
        
        smooth_path = Route(spline_points=[self.spline_points[0]])
        
        for i in range(len(self.spline_points) - 1):
            next_point = self.spline_points[i + 1]
            current_yaw, next_yaw = self.spline_points[i].yaw, next_point.yaw
            dyaw = next_yaw - current_yaw

            while dyaw >= math.pi / 2.0:
                next_yaw -= math.pi * 2.0
                dyaw = next_yaw - current_yaw

            while dyaw <= -math.pi / 2.0:
                next_yaw += math.pi * 2.0
                dyaw = next_yaw - current_yaw
            
            soft_point = SplinePoint(
                x=next_point.x,
                y=next_point.y,
                yaw=next_yaw,
                curvature=next_point.curvature)
            smooth_path.add_point(soft_point)
        
        return smooth_path
