from plot_env import EnvPlot
import numpy as np
import math

import PSO_MPC_module
import PSO_MPC_module2

class ManualPlot(EnvPlot):
    def __init__(self, maxLen=None):
        super().__init__(maxLen)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.tgtPos = np.zeros(2)

    def onclick(self, event):
        print(event.xdata, event.ydata)
        self.tgtPos = np.array([event.xdata, event.ydata])

def update_action(action, diff_action, sec):
    v_scale     = 1.0
    # theta_scale = math.pi / 2.0
    theta_scale = math.pi
    w_scale     = math.pi / 4.0

    limit_v     = v_scale     * sec
    limit_theta = theta_scale * sec
    limit_w     = w_scale     * sec

    delta_v     = diff_action[0] * limit_v
    delta_theta = diff_action[1] * limit_theta
    delta_w     = diff_action[2] * limit_w

    v     = action[0] + delta_v
    theta = action[1] + delta_theta
    w     = action[2] + delta_w

    if v < -v_scale:
        v = -v_scale
    elif v > v_scale:
        v = v_scale

    if theta > math.pi:
        theta -= 2*math.pi
    elif theta < -math.pi:
        theta += 2*math.pi
    # if theta > theta_scale:
    #     theta = 2*theta_scale
    # elif theta < -theta_scale:
    #     theta = 2*theta_scale

    if w < -w_scale:
        w = -w_scale
    elif w > w_scale:
        w = w_scale

    action[0] = v
    action[1] = theta
    action[2] = w

def update_X(X, action, sec):
    dtheta = X[2] + action[1]
    sin = math.sin(dtheta)
    cos = math.cos(dtheta)

    vx = action[0] * -sin
    vy = action[0] *  cos
    w  = action[2]

    X[0] = X[0] + vx*sec
    X[1] = X[1] + vy*sec
    X[2] = X[2] +  w*sec

def calc_tgt_local(X, tgtPos):
    dx = tgtPos[0] - X[0]
    dy = tgtPos[1] - X[1]

    l = math.sqrt(dx**2 + dy**2)
    t = math.atan2(dy, dx)
    # print(t)
    t_local = t - X[2]
    dx_local = l*math.cos(t_local)
    dy_local = l*math.sin(t_local)

    return [dx_local, dy_local]

def test1():
    X = np.zeros(3)
    # X[2] = math.pi/2.0
    action = np.zeros(3)
    sec = 0.1

    manualPlot = ManualPlot(maxLen=4)

    while True:
        tgt_local = calc_tgt_local(X, manualPlot.tgtPos)
        global_best_positions, global_score = PSO_MPC_module2.calc_best(
        # global_best_positions, grobal_scores = PSO_MPC_module.calc_best(
            N=150, 
            time_step=10, 
            sec=sec, 
            personal_cost=1.0, 
            global_cost=1.0, 
            max_iter=10, 
            tgtPos=tgt_local,
            action0=action
            )

        update_action(action, global_best_positions[0], sec)
        update_X(X, action, sec)
        # print(tgt_local, X, action)
        print(action)
        # print(global_score)
        # print(tgt_local)
        manualPlot.clear()
        manualPlot.drawMap()
        manualPlot.draw_points(np.array([manualPlot.tgtPos]), psize=10)
        manualPlot.drawRobot(X[:2], X[2])
        # manualPlot.drawAction([-3, 2], 0.7+np.random.rand()/3, theta+np.random.rand()*0.5, 2.0*np.random.rand()-1.0)
        manualPlot.update(0.001)

def test2():
    X = np.zeros(3)
    # X[2] = math.pi/2.0
    action = np.zeros(3)
    sec = 0.1

    manualPlot = ManualPlot(maxLen=4)

    tgt_local = calc_tgt_local(X, manualPlot.tgtPos)
    global_best_positions, grobal_score = PSO_MPC_module2.calc_best2(
        N=100, 
        time_step=10, 
        sec=sec, 
        personal_cost=1.0, 
        global_cost=1.0, 
        max_iter=10, 
        tgtPos=tgt_local,
        action0=action
        )

    while True:
        tgt_local = calc_tgt_local(X, manualPlot.tgtPos)
        global_best_positions, grobal_score = PSO_MPC_module2.calc_best2(
        N=100, 
        time_step=10, 
        sec=sec, 
        personal_cost=1.0, 
        global_cost=1.0, 
        max_iter=10, 
        tgtPos=tgt_local,
        action0=action,
        global_best_positions=global_best_positions,
        grobal_score=grobal_score
        )

        update_action(action, global_best_positions[0], sec)
        update_X(X, action, sec)
        # print(tgt_local, X, action)
        print(action)
        # print(global_score)
        # print(tgt_local)
        manualPlot.clear()
        manualPlot.drawMap()
        manualPlot.draw_points(np.array([manualPlot.tgtPos]), psize=10)
        manualPlot.drawRobot(X[:2], X[2])
        # manualPlot.drawAction([-3, 2], 0.7+np.random.rand()/3, theta+np.random.rand()*0.5, 2.0*np.random.rand()-1.0)
        manualPlot.update(0.001)

if __name__ == '__main__':

    test1()
    # test2()