from state import State

class PathManager:
    def __init__(self):
        pass

    def get_starting_point(self) -> tuple[Spline, State]:
        pass

    def get_spline(self, current_spline: Spline, current_state: State) -> Spline:
        pass
    
class StaticPathManager(PathManager):
    def __init__(self, spline: Spline):
        super().__init__()
        self._spline = spline

    def get_spline(self, current_state: State) -> Spline:
        return self._spline