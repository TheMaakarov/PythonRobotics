import numpy as np
import numpy.typing as npt
import math
import cvxpy
from typing import Optional
from collections.abc import Callable

from mpc_config import MpcConfig
from geometry import State, SplinePoint, Route, pi_2_pi, get_nparray_from_matrix
from enums import LogLevel
from movement import MpcAction, MovementHistory
from path_provider import PathProvider
from mpc_plotter import plot_car_state, plot_err_stats

NpArr = npt.NDArray[np.float64]

class MPCController:
    
    def __init__(
        self,
        mpc_config: MpcConfig,
        path_provider: PathProvider):
        
        self._log_action: Optional[Callable[[str, LogLevel], None]] = None
        self._mpc_config = mpc_config
        self._path_provider = path_provider
        
        # We save a reference of the actual path to avoid losing resolution when smoothing it
        self._raw_route = Route(spline_points=[])
        self._smooth_route = Route(spline_points=[])
        self._history: MovementHistory = None # type: ignore
        
    def bind_log_action(self, log_action: Callable[[str, LogLevel], None]):
        self._log_action = log_action
        
    def log(self, message: str, log_level: LogLevel):
        if self._log_action is None:
            return
        self._log_action(message, log_level)
    
    def next_action(self, car_state: Optional[State]) -> tuple[State, Optional[MpcAction]]:
        
        first_state = self._initialize_mpc()
        if not car_state:
            car_state = first_state
        
        if not car_state:
            self.log('Unexpected case without any initial state', LogLevel.Error)
            return State(0, 0), None
            
        if not self._history:
            return car_state, None
        
        spline_point = self._path_provider.get_next_spline_point(car_state)
           
        if spline_point is not None:
            self._process_new_point(spline_point)
        
        next_action_values = self._iterate_next_action(car_state)
        predicted_state = self._history.get_state_history()[-1]
        return predicted_state, next_action_values

    def _initialize_mpc(self) -> Optional[State]:
        
        if self._history:
            return None
        
        first_point = self._path_provider.get_first_spline_point()

        if first_point is None:
            return None
        
        first_state = State(first_point.x, first_point.y, first_point.yaw, v=0.0)

        # initial yaw compensation
        if first_state.yaw - first_point.yaw >= math.pi:
            first_state.yaw -= math.pi * 2.0
        elif first_state.yaw - first_point.yaw <= -math.pi:
            first_state.yaw += math.pi * 2.0

        self._history = MovementHistory(first_state)
        self._process_new_points([first_point])
        return first_state

    def _process_new_point(self, spline_point: SplinePoint):
        self._raw_route.add_point(spline_point)
        self._smooth_route = self._raw_route.get_smooth_path()
        self.log(f'New point added: {spline_point}', LogLevel.Info)
        self._speed_profile = self._calc_path_speed_profile(self._smooth_route)
        
    def _process_new_points(self, spline_points: list[SplinePoint]):
        for spline_point in spline_points:
            self._raw_route.add_point(spline_point)
            self._smooth_route = self._raw_route.get_smooth_path()
            self.log(f'New point added: {spline_point}', LogLevel.Debug)
        
        self._speed_profile = self._calc_path_speed_profile(self._smooth_route)
        
    def _iterate_next_action(self, state: State) -> Optional[MpcAction]:
        
        def build_last_predicted_actions() -> tuple[NpArr, NpArr]:
            
            oa: NpArr
            odelta: NpArr

            last_actions: list[MpcAction] = self._history.last_prediction
            horizon_length = self._mpc_config.algorithm.T
            
            if not last_actions:
                oa = np.zeros(horizon_length, dtype=np.float64)
                odelta = np.zeros(horizon_length, dtype=np.float64)
                return oa, odelta
            
            oa =     np.fromiter((a.acceleration   for a in last_actions), dtype=np.float64)
            odelta = np.fromiter((a.steering_angle for a in last_actions), dtype=np.float64)
            return oa, odelta
        
        oa, odelta = build_last_predicted_actions()
        
        dl = self._path_provider.course_tick
        prev_index = self._history.last_target_index
        xref, target_ind, dref = self._calc_route_ref_trajectory(
                state, dl, prev_index)
        self._history.last_target_index = target_ind
        
        mpc_results = self._iterative_linear_mpc_control(
            xref, state, dref, oa, odelta)
        
        if not mpc_results:
            return None
        
        oa, odelta, ox, oy, _, _ = mpc_results

        prediction: list[MpcAction] = []
        for (acc, steer) in zip(oa, odelta):
            prediction.append(MpcAction(acc, steer))
                    
        top_action = prediction[0]
        state = self._update_state(state, top_action)

        self._history.add_state(state, dl)
        self._history.last_prediction = prediction
        
        if self._check_goal(state, target_ind):
            self._history.reached_goal = True
            self.log("Goal reached!", LogLevel.Info)

        v_config = self._mpc_config.vehicle
        s_config = self._mpc_config.simulation
        state_history = self._history.get_state_history()
        if s_config.SHOW_ANIMATION: # pragma: no cover
            course_x, course_y, _, _ = self._smooth_route.get_spline_data()
            states_x, states_y = zip(*[(st.x, st.y) for st in state_history])
            time = self._history.ticked_time
            plot_car_state(s_config, v_config, course_x, course_y, state, time, states_x, states_y, target_ind, xref, ox, oy, top_action.steering_angle)
        
        if s_config.SHOW_ERR_STATS:
            plot_err_stats(self._history, self._smooth_route, s_config)

        return top_action

    def _iterative_linear_mpc_control(self, xref: NpArr, state: State, dref: NpArr, oa: NpArr, od: NpArr) \
        -> Optional[tuple[NpArr, NpArr, Optional[NpArr], Optional[NpArr], Optional[NpArr], Optional[NpArr]]]:
        """
        MPC control with updating operational point iteratively
        """
        max_iter, du_th = self._mpc_config.algorithm.MAX_ITER, self._mpc_config.algorithm.DU_TH
        
        ox: Optional[NpArr] = None
        oy: Optional[NpArr] = None
        oyaw: Optional[NpArr] = None
        ov: Optional[NpArr] = None

        for _ in range(max_iter):
            xbar = self._predict_motion(state, oa, od, xref)
            poa, pod = oa.copy(), od.copy()
            mpc_result = self._linear_mpc_control(xref, xbar, state, dref)
            if not mpc_result:
                return None
            oa, od, ox, oy, oyaw, ov = mpc_result
            
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= du_th:
                break
        else:
            self.log("Iterative is max iter", LogLevel.Warn)

        return oa, od, ox, oy, oyaw, ov

    # TODO: perform typing
    def _linear_mpc_control(self, xref: NpArr, xbar: NpArr, state: State, dref: NpArr) \
        -> Optional[tuple[NpArr, NpArr, NpArr, NpArr, NpArr, NpArr]]:
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        state: initial state
        dref: reference steer angle
        """

        alg_conf = self._mpc_config.algorithm
        vehicle_conf = self._mpc_config.vehicle
        horizon_length = alg_conf.T
        
        x = cvxpy.Variable((alg_conf.NX, horizon_length + 1))
        u = cvxpy.Variable((alg_conf.NU, horizon_length))

        cost = 0.0
        constraints = []

        for t in range(horizon_length):
            cost += cvxpy.quad_form(u[:, t], alg_conf.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], alg_conf.Q)

            A, B, C = self._get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (horizon_length - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], alg_conf.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= vehicle_conf.MAX_DSTEER * alg_conf.DT]

        cost += cvxpy.quad_form(xref[:, horizon_length] - x[:, horizon_length], alg_conf.Qf)

        constraints += [x[:, 0] == state.components]
        constraints += [x[2, :] <= vehicle_conf.MAX_SPEED]
        constraints += [x[2, :] >= vehicle_conf.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= vehicle_conf.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= vehicle_conf.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :]) # type: ignore
            oy = get_nparray_from_matrix(x.value[1, :]) # type: ignore
            ov = get_nparray_from_matrix(x.value[2, :]) # type: ignore
            oyaw = get_nparray_from_matrix(x.value[3, :]) # type: ignore
            oa = get_nparray_from_matrix(u.value[0, :]) # type: ignore
            odelta = get_nparray_from_matrix(u.value[1, :]) # type: ignore

        else:
            self.log("Cannot solve MPC..", LogLevel.Error)
            return None

        return oa, odelta, ox, oy, oyaw, ov

    def _get_linear_model_matrix(self, v: float, phi: float, delta: float) -> tuple[NpArr, NpArr, NpArr]:
        dt = self._mpc_config.algorithm.DT
        wb = self._mpc_config.vehicle.WB
        nx = self._mpc_config.algorithm.NX
        nu = self._mpc_config.algorithm.NU

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
    
    def _update_state(self, state: State, mpc_action: MpcAction) -> State:

        min_speed = self._mpc_config.vehicle.MIN_SPEED
        max_speed = self._mpc_config.vehicle.MAX_SPEED
        max_steer = self._mpc_config.vehicle.MAX_STEER
        wb = self._mpc_config.vehicle.WB
        dt = self._mpc_config.algorithm.DT
        
        # input check
        delta = max(-max_steer, min(mpc_action.steering_angle, max_steer))

        x = state.x + state.v * math.cos(state.yaw) * dt
        y = state.y + state.v * math.sin(state.yaw) * dt
        yaw = state.yaw + state.v / wb * math.tan(delta) * dt
        v = state.v + mpc_action.acceleration * dt
        v = max(min_speed, min(v, max_speed))

        new_state = State(x, y, yaw, v)
        return new_state
    
    def _calc_nearest_index(self, state: State, cx: list[float], cy: list[float], cyaw: list[float], pind: int) -> tuple[int, float]:
        """
        Calculate the nearest index and distance from the current state to the reference path.

        Returns:
            ind (int): The index of the nearest point on the path.
            mind (float): The distance to the nearest point on the path.
        """
        
        n_ind_search = self._mpc_config.algorithm.N_IND_SEARCH
        
        dx = [state.x - icx for icx in cx[pind:(pind + n_ind_search)]]
        dy = [state.y - icy for icy in cy[pind:(pind + n_ind_search)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def _predict_motion(self, state: State, oa: NpArr, od: NpArr, xref: NpArr) \
        -> NpArr:
            
        horiz_length = self._mpc_config.algorithm.T
        xbar = xref * 0.0

        x0 = state.components
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        for (ai, di, i) in zip(oa, od, range(1, horiz_length + 1)):
            mpc_action = MpcAction(ai, di)
            state = self._update_state(state, mpc_action)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar
    
    def _calc_route_ref_trajectory(self, state: State, dl: float, pind: int) \
        -> tuple[NpArr, int, NpArr]:
            
        cx, cy, cyaw, _ = self._smooth_route.get_spline_data()
        sp = self._speed_profile
        return self._calc_ref_trajectory(state, cx, cy, cyaw, sp, dl, pind)

    def _calc_ref_trajectory(self, state: State, cx: list[float], cy: list[float], cyaw: list[float], sp: list[float], dl: float, pind: int) \
        -> tuple[NpArr, int, NpArr]:
        
        algorithm_config = self._mpc_config.algorithm

        xref = np.zeros((algorithm_config.NX, algorithm_config.T + 1))
        dref = np.zeros((1, algorithm_config.T + 1))
        ncourse = len(cx)

        ind, _ = self._calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(algorithm_config.T + 1):
            travel += abs(state.v) * algorithm_config.DT
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref

    def _check_goal(self, state: State, tind: int) -> bool:
        
        goal = self._smooth_route.spline_points[-1]
        
        nind = len(self._history.get_state_history())
        
        goal_dis = self._mpc_config.algorithm.GOAL_DIS
        stop_speed = self._mpc_config.algorithm.STOP_SPEED
        
        # check goal
        dx = state.x - goal.x
        dy = state.y - goal.y
        d = math.hypot(dx, dy)

        isgoal = (d <= goal_dis)

        if abs(tind - nind) >= 5:
            isgoal = False

        isstop = (abs(state.v) <= stop_speed)

        return isgoal and isstop

    def _calc_path_speed_profile(self, route: Route) -> list[float]:
        cx, cy, cyaw, _ = route.get_spline_data()
        return self._calc_speed_profile(cx, cy, cyaw)

    def _calc_speed_profile(self, cx: list[float], cy: list[float], cyaw: list[float]):
        target_speed = self._mpc_config.algorithm.TARGET_SPEED
        
        speed_profile = [target_speed] * len(cx)
        direction = 1.0  # forward

        # Set stop point
        for i in range(len(cx) - 1):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]

            move_direction = math.atan2(dy, dx)

            if dx != 0.0 and dy != 0.0:
                dangle = abs(pi_2_pi(move_direction - cyaw[i]))
                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

        speed_profile[-1] = 0.0

        return speed_profile