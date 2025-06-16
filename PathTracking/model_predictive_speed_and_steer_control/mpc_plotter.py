import matplotlib.pyplot as plt
import numpy as np

import math

from mpc_config import VehicleConfig

def init_plot(cx, cy, t, x, y, v):
    plt.close("all")
    plt.subplots()
    plt.plot(cx, cy, "-r", label="spline")
    plt.plot(x, y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots()
    plt.plot(t, v, "-r", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [kmh]")

    plt.show()
    
 
def plot_state(vehicle_config: VehicleConfig, cx, cy, state, time, x, y, target_ind, xref, ox, oy, di):
    plt.cla()
            # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
    if ox is not None:
        plt.plot(ox, oy, "xr", label="MPC")
    plt.plot(cx, cy, "-r", label="course")
    plt.plot(x, y, "ob", label="trajectory")
    plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    plot_car(vehicle_config, state.x, state.y, state.yaw, steer=di)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
    plt.pause(0.0001)


def plot_car(vehicle_config: VehicleConfig, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    back_to_wheel, length, width, wheel_len, wheel_width, tread, wb = \
        vehicle_config.BACKTOWHEEL, vehicle_config.LENGTH, vehicle_config.WIDTH, vehicle_config.WHEEL_LEN, vehicle_config.WHEEL_WIDTH, vehicle_config.TREAD, vehicle_config.WB
    
    outline = np.array([[-back_to_wheel, (length - back_to_wheel), (length - back_to_wheel), -back_to_wheel, -back_to_wheel],
                        [width / 2, width / 2, - width / 2, -width / 2, width / 2]])

    fr_wheel = np.array([[wheel_len, -wheel_len, -wheel_len, wheel_len, wheel_len],
                         [-wheel_width - tread, -wheel_width - tread, wheel_width - tread, wheel_width - tread, -wheel_width - tread]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += wb
    fl_wheel[0, :] += wb

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

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

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")
