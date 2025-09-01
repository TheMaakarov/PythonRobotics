import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from geometry import State, Route
from mpc_config import VehicleConfig, SimulationConfig
from movement import MovementHistory

# Persistent figure and axes so both functions draw in the same window
_FIG: Optional[Figure] = None
_AX_CAR: Optional[Axes] = None
_AX_ERR: Optional[Axes] = None

def plot_car_state(sim_config: SimulationConfig, vehicle_config: VehicleConfig, cx, cy, state: State, time, x, y, target_ind, xref, ox, oy, di):
    if not sim_config.SHOW_ANIMATION:
        return
    
    fig, ax_car, _ = _ensure_fig()
    assert fig is not None and ax_car is not None
    ax_car.clear()
    if ox is not None:
        ax_car.plot(ox, oy, "xr", label="MPC")
    ax_car.plot(cx, cy, "-r", label="course")
    ax_car.plot(x, y, "ob", label="trajectory")
    ax_car.plot(cx[target_ind], cy[target_ind], "xb", label="target")
    ax_car.plot(xref[0, :], xref[1, :], "xk", label="xref")
    _plot_car(vehicle_config, ax_car, state.x, state.y, state.yaw, steer=di)
    ax_car.set_aspect('equal', adjustable='box')
    ax_car.grid(True)
    ax_car.set_title("Time[s]:" + str(round(time, 2)) +
                     ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
    ax_car.legend()
    
    fig.canvas.draw_idle()
    plt.pause(0.001)
    
def plot_err_stats(status_history: MovementHistory, actual_route: Route, sim_config: SimulationConfig):
    if not sim_config.SHOW_ERR_STATS:
        return
    
    # Prepare data
    states = status_history.get_state_history()
    if len(states) == 0:
        print("No states in status_history to plot.")
        return

    if actual_route is None or len(actual_route.spline_points) == 0:
        print("No actual_route provided to compare against.")
        return

    sx = np.array([s.x for s in states])
    sy = np.array([s.y for s in states])

    rx = np.array([p.x for p in actual_route.spline_points])
    ry = np.array([p.y for p in actual_route.spline_points])

    if len(status_history.recorded_times) == len(states):
        times = np.array(status_history.recorded_times)
    else:
        times = np.arange(len(states))

    # For each recorded state find nearest point on route and compute deviation
    deviations = np.zeros(len(states))
    nearest_idx = np.zeros(len(states), dtype=int)
    for i, (x, y) in enumerate(zip(sx, sy)):
        d2 = (rx - x) ** 2 + (ry - y) ** 2
        idx = int(np.argmin(d2))
        nearest_idx[i] = idx
        deviations[i] = float(np.sqrt(d2[idx]))

    # Use the persistent figure axes: top = XY (car), bottom = deviation vs time
    fig, ax_car, ax_err = _ensure_fig()
    assert fig is not None and ax_car is not None and ax_err is not None

    # Remove previously plotted route/state/connection artists from car axis to avoid overplotting
    remove_labels = {'route', 'recorded states', 'nearest_conn'}
    # remove Line2D artists by label
    for line in list(ax_car.get_lines()):
        lbl = getattr(line, 'get_label', lambda: None)()
        if lbl in remove_labels:
            try:
                line.remove()
            except Exception:
                pass

    # Update only the error axis for deviation vs time
    ax_err.clear()
    ax_err.plot(times, deviations, '-r', label='deviation')
    ax_err.set_xlabel('Time [s]')
    ax_err.set_ylabel('Deviation [m]')
    ax_err.set_title('Deviation of recorded states to nearest route point')
    ax_err.grid(True)
    ax_err.legend()

    fig.canvas.draw_idle()
    # non-blocking update
    plt.pause(0.001)

def _plot_car(car: VehicleConfig, ax, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):

    # defensively handle None steer values
    if steer is None:
        steer = 0.0

    outline = np.matrix([[-car.BACKTOWHEEL, (car.LENGTH - car.BACKTOWHEEL), (car.LENGTH - car.BACKTOWHEEL), -car.BACKTOWHEEL, -car.BACKTOWHEEL],
                         [car.WIDTH / 2, car.WIDTH / 2, - car.WIDTH / 2, -car.WIDTH / 2, car.WIDTH / 2]])

    fr_wheel = np.matrix([[car.WHEEL_LEN, -car.WHEEL_LEN, -car.WHEEL_LEN, car.WHEEL_LEN, car.WHEEL_LEN],
                          [-car.WHEEL_WIDTH - car.TREAD, -car.WHEEL_WIDTH - car.TREAD, car.WHEEL_WIDTH - car.TREAD, car.WHEEL_WIDTH - car.TREAD, -car.WHEEL_WIDTH - car.TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
                      [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.matrix([[math.cos(steer), math.sin(steer)],
                      [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T * Rot2).T
    fl_wheel = (fl_wheel.T * Rot2).T
    fr_wheel[0, :] += car.WB
    fl_wheel[0, :] += car.WB

    fr_wheel = (fr_wheel.T * Rot1).T
    fl_wheel = (fl_wheel.T * Rot1).T

    outline = (outline.T * Rot1).T
    rr_wheel = (rr_wheel.T * Rot1).T
    rl_wheel = (rl_wheel.T * Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    ax.plot(np.array(outline[0, :]).flatten(),
        np.array(outline[1, :]).flatten(), truckcolor)
    ax.plot(np.array(fr_wheel[0, :]).flatten(),
        np.array(fr_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(rr_wheel[0, :]).flatten(),
        np.array(rr_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(fl_wheel[0, :]).flatten(),
        np.array(fl_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(rl_wheel[0, :]).flatten(),
        np.array(rl_wheel[1, :]).flatten(), truckcolor)
    ax.plot(x, y, "*")
    # ensure axis limits include the newly drawn car
    try:
        ax.relim()
        ax.autoscale_view()
    except Exception:
        pass

def _ensure_fig():
    global _FIG, _AX_CAR, _AX_ERR
    if _FIG is None or not plt.fignum_exists(_FIG.number):
        plt.ion()
        fig, (ax_car, ax_err) = plt.subplots(2, 1, figsize=(9, 10))
        # set globals after creation
        _FIG = fig
        _AX_CAR = ax_car
        _AX_ERR = ax_err
        # manager may be None depending on backend
        mgr = getattr(_FIG.canvas, 'manager', None)
        if mgr is not None:
            try:
                mgr.set_window_title('Path Tracking')
            except Exception:
                pass
        plt.tight_layout()
    return _FIG, _AX_CAR, _AX_ERR