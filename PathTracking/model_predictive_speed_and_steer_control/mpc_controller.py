import numpy as np
import math
import cvxpy

from mpc_config import AlgorithmConfig, VehicleConfig, SimulationConfig, default_algorithm_config, default_vehicle_config, default_simulation_config
from state import State
from path_manager import PathManager, StaticPathManager
from mpc_plotter import init_plot, plot_state
from model_predictive_speed_and_steer_control import pi_2_pi, get_nparray_from_matrix, smooth_yaw

class MPCController:    
    def __init__(
        self,
        path_manager: PathManager = StaticPathManager(),
        mpc_config: AlgorithmConfig = default_algorithm_config,
        vehicle_config: VehicleConfig = default_vehicle_config,
        simulation_config: SimulationConfig = default_simulation_config):
        
        self._path_manager = path_manager
        self._mpc_config = mpc_config
        self._vehicle_config = vehicle_config
        self._simulation_config = simulation_config


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
    
    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        n_ind_search = self._mpc_config.N_IND_SEARCH
        
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
    
    def predict_motion(self, x0, oa, od, xref):
        horiz_length = self._mpc_config.T
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, horiz_length + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar
    
    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """
        MPC control with updating operational point iteratively
        """
        max_iter, du_th = self._mpc_config.MAX_ITER, self._mpc_config.DU_TH
        
        ox, oy, oyaw, ov = None, None, None, None

        horizon_length = self._mpc_config.T
        if oa is None or od is None:
            oa = [0.0] * horizon_length
            od = [0.0] * horizon_length

        for i in range(max_iter):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= du_th:
                break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov
    
    def linear_mpc_control(self, xref, xbar, x0, dref):
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        alg_conf = self._mpc_config
        vehicle_conf = self._vehicle_config

        x = cvxpy.Variable((alg_conf.NX, alg_conf.T + 1))
        u = cvxpy.Variable((alg_conf.NU, alg_conf.T))

        cost = 0.0
        constraints = []

        for t in range(alg_conf.T):
            cost += cvxpy.quad_form(u[:, t], alg_conf.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], alg_conf.Q)

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (alg_conf.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], alg_conf.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                vehicle_conf.MAX_DSTEER * alg_conf.DT]

        cost += cvxpy.quad_form(xref[:, alg_conf.T] - x[:, alg_conf.T], alg_conf.Qf)

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= vehicle_conf.MAX_SPEED]
        constraints += [x[2, :] >= vehicle_conf.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= vehicle_conf.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= vehicle_conf.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov
    
    def calc_ref_trajectory(self, state, cx, cy, cyaw, ck, sp, dl, pind):
        conf = self._mpc_config

        xref = np.zeros((conf.NX, conf.T + 1))
        dref = np.zeros((1, conf.T + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(conf.T + 1):
            travel += abs(state.v) * conf.DT
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

    def check_goal(self, state, goal, tind, nind):
        
        goal_dis = self._mpc_config.GOAL_DIS
        stop_speed = self._mpc_config.STOP_SPEED
        
        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        d = math.hypot(dx, dy)

        isgoal = (d <= goal_dis)

        if abs(tind - nind) >= 5:
            isgoal = False

        isstop = (abs(state.v) <= stop_speed)

        if isgoal and isstop:
            return True

        return False
    
    def do_simulation(self, cx, cy, cyaw, ck, sp, dl, initial_state: State):
        """
        Simulation

        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]

        """
        
        conf = self._mpc_config

        goal = [cx[-1], cy[-1]]

        state = initial_state

        # initial yaw compensation
        if state.yaw - cyaw[0] >= math.pi:
            state.yaw -= math.pi * 2.0
        elif state.yaw - cyaw[0] <= -math.pi:
            state.yaw += math.pi * 2.0

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]
        target_ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, 0)

        odelta, oa = None, None

        cyaw = smooth_yaw(cyaw)

        while conf.MAX_TIME >= time:
            xref, target_ind, dref = self.calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl, target_ind)

            x0 = [state.x, state.y, state.v, state.yaw]  # current state

            oa, odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
                xref, x0, dref, oa, odelta)

            steer, acceleration = 0.0, 0.0
            if odelta is not None:
                steer, acceleration = odelta[0], oa[0]
                state = self.update_state(state, acceleration, steer)

            time = time + conf.DT

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            t.append(time)
            d.append(steer)
            a.append(acceleration)

            if self.check_goal(state, goal, target_ind, len(cx)):
                print("Goal")
                break

            if self._simulation_config.SHOW_ANIMATION:  # pragma: no cover
                plot_state(cx, cy, state, time, x, y, target_ind, xref, ox, oy, steer)

        return t, x, y, yaw, v, d, a
