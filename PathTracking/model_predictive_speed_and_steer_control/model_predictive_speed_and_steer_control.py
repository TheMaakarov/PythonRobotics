"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import time
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from PathPlanning.CubicSpline import cubic_spline_planner

from mpc_controller import MPCController
import mpc_config as conf
from path_provider import StaticPathProvider
from geometry import Route, State, SplinePoint


def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck


def main():
    """
    # cx, cy, cyaw, ck = get_straight_course(dl)
    # cx, cy, cyaw, ck = get_straight_course2(dl)
    # cx, cy, cyaw, ck = get_straight_course3(dl)
    # cx, cy, cyaw, ck = get_forward_course(dl)
    cx, cy, cyaw, ck = get_switch_back_course(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if show_animation:  # pragma: no cover
        init_plot(cx, cy, t, x, y, v)
    """
    print(__file__ + " start!!")
    start = time.time()

    dl = 1.0  # course tick
    alg_conf = conf.AlgorithmConfig()
    vehicle_conf = conf.VehicleConfig()
    sim_conf = conf.SimulationConfig()
    mpc_config = conf.MpcConfig(
        alg_conf,
        vehicle_conf,
        sim_conf)
    
    cx, cy, cyaw, ck = get_forward_course(dl)
    spline_points: list[SplinePoint] = []
    for x, y, yaw, k in zip(cx, cy, cyaw, ck):
        spline_points.append(SplinePoint(x, y, yaw, k))
        
    route = Route(list(spline_points))
    path_provider = StaticPathProvider(route, dl)
    initial_point = path_provider.get_first_spline_point()
    if not initial_point:
        print('ERROR: No initial point')
        return
    
    state = State(initial_point.x, initial_point.y, initial_point.yaw, 0.0)
    mpc_controller = MPCController(
        mpc_config,
        path_provider)
    
    rt = 0
    while time.time() - start < 1000000 and rt < 10:
        state, action = mpc_controller.next_action(state)
        if action:
            rt = 0
            #print(f'ACTION!: {action}')
        else:
            rt += 1
            
    print('OUT!')

def main2():
    """
    print(__file__ + " start!!")
    start = time.time()

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if show_animation:  # pragma: no cover
        init_plot(cx, cy, t, x, y, v)
    """
    pass

if __name__ == '__main__':
    main()
    # main2()
