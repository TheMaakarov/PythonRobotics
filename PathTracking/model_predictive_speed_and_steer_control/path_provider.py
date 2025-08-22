from geometry import State, SplinePoint, Route

class PathProvider:
    def __init__(self, course_tick: float = 1.0):
        self._course_tick = course_tick

    @property
    def course_tick(self) -> float:
        return self._course_tick

    def get_first_spline_point(self) -> SplinePoint | None:
        return None

    def get_next_spline_point(self, car_state: State) -> SplinePoint | None:
        return None
    
class StaticPathProvider(PathProvider):
    def __init__(self, path: Route, course_tick: float = 1.0):
        super().__init__(course_tick)
        self._last_provided_point = 0
        self._path = path
        
    def get_first_spline_point(self) -> SplinePoint | None:
        return self._path.spline_points[0] if self._path.spline_points else None

    def get_next_spline_point(self, car_state: State) -> SplinePoint | None:
        if self._last_provided_point >= len(self._path.spline_points):
            return None
        
        next_point = self._path.spline_points[self._last_provided_point]
        self._last_provided_point += 1
        return next_point