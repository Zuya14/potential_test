import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optim_

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

from plot_env import EnvPlot

class PotentialField:

    def __init__(self, pos):
        self.pos = pos

    def calc(self, x, g, max=100.0, min=-100.0):
        temp = []

        for p in self.pos:
            # temp.append(self.calc_cost(x, p, w=0.1).unsqueeze(0))
            temp.append(self.calc_RepulsivePotential(x, p, w=0.05, d_thr=0.30+0.25).unsqueeze(0))
            # temp.append(torch.clamp(self.calc_RepulsivePotential(x, p, w=0.05, d_thr=0.30+0.25).unsqueeze(0), max=max, min=min))

        # temp.append(self.calc_cost(x, g, w=-1.0).view(1).unsqueeze(0))
        # temp.append(self.calc_cost2(x, g, w=1.0).view(1).unsqueeze(0))
        # temp.append(self.calc_AttractivePotential(x, g, w=1.0, d_thr=1.0).unsqueeze(0))

        # # return torch.clamp(torch.sum(torch.cat(temp), dim=0), max=max, min=min)
        # return torch.sum(torch.cat(temp), dim=0)

        # return torch.clamp(torch.sum(torch.cat(temp), dim=0), max=max, min=min) + self.calc_AttractivePotential(x, g, w=1.0, d_thr=0.1)
        # return torch.sum(torch.cat(temp), dim=0)
        return torch.clamp(torch.sum(torch.cat(temp), dim=0), max=max, min=min) + self.calc_cost2(x, g, w=1.0)

    def calc_cost(self, x, p, w=1.0):
        return w * torch.reciprocal(torch.norm((x[0] - p[0])**2 + (x[1] - p[1])**2) +1e-2)

    def calc_cost2(self, x, p, w=1.0):
        return w * torch.norm((x[0] - p[0])**2 + (x[1] - p[1])**2)

    def calc_AttractivePotential(self, x, p, w=1.0, d_thr=1.0):
        d = torch.norm((x[0] - p[0])**2 + (x[1] - p[1])**2).view(1)

        # if d.item() > d_thr:
        #     return d_thr * w * d - w / 2.0 * d_thr**2
        # else:
        #     return w / 2.0 * d**2

        return torch.where(
            d > d_thr,
            d_thr * w * d - w / 2.0 * d_thr**2,
            w / 2.0 * d**2
            ).view(1)

    def calc_RepulsivePotential(self, x, p, w=1.0, d_thr=1.0):
        d = torch.norm((x[0] - p[0])**2 + (x[1] - p[1])**2)

        # if d.item() < d_thr:
        #     return torch.zeros(1)
        # else:
        #     return torch.tensor(w / 2.0 * (1.0/d.item() + (1.0/d_thr))**2).view(1)

        return torch.where(
            d > d_thr,
            d*0.0,
            w / 2.0 * (1.0/d + (1.0/d_thr))**2
            ).view(1)

class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.m = torch.zeros(2)
        self.v = torch.zeros(2)
        self.t = 1

    def update(self, grad):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        self.t += 1

        return self.alpha * m_hat / torch.sqrt(v_hat + self.epsilon)

class ManualPlot(EnvPlot):
    def __init__(self, maxLen=None):
        super().__init__(maxLen)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.tgtPos = np.zeros(2)
        self.click_flag = False

    def onclick(self, event):
        print(event.xdata, event.ydata)
        self.tgtPos = np.array([event.xdata, event.ydata])
        self.click_flag = True

    def update(self, interval=0.01):
        super().update(interval)
        self.click_flag = False

def test1():
    eta = 0.01

    # adam = Adam(alpha=0.01)

    x = torch.zeros(2, requires_grad=True)
    x.requires_grad = True 

    # optimizer = optim.SGD([x], lr=0.01)
    # optimizer = optim.Adam([x], lr=0.1)
    # optimizer = optim.Adam([x], lr=0.1, betas=(0.9, 0.999))
    # optimizer = optim.Adam([x], lr=0.01, betas=(0.0, 0.0))
    optimizer = optim_.AdaBelief([x], lr=0.1)
    # optimizer = optim_.AdaBelief([x], lr=0.1, betas=(0.9, 0.9))

    obst_pos = [
        # [-0.9+0.15,  0.5+0.5],
        # [ 0.6+0.15,  0.5+0.5],
        # [ 2.1+0.15,  0.5+0.5],
        # [-0.9+0.15, -2.5+0.5],
        # [ 0.6+0.15, -2.5+0.5],
        # [ 2.1+0.15, -2.5+0.5],

        # [-0.9+0.15,  0.5+1.0],
        # [ 0.6+0.15,  0.5+1.0],
        # [ 2.1+0.15,  0.5+1.0],
        # [-0.9+0.15, -2.5+1.0],
        # [ 0.6+0.15, -2.5+1.0],
        # [ 2.1+0.15, -2.5+1.0],

        # [-0.9+0.15,  0.5+1.5],
        # [ 0.6+0.15,  0.5+1.5],
        # [ 2.1+0.15,  0.5+1.5],
        # [-0.9+0.15, -2.5+1.5],
        # [ 0.6+0.15, -2.5+1.5],
        # [ 2.1+0.15, -2.5+1.5],

        # [-0.9,  0.5],
        # [ 0.6,  0.5],
        # [ 2.1,  0.5],
        # [-0.9, -2.5],
        # [ 0.6, -2.5],
        # [ 2.1, -2.5],

        # [-0.9,  0.5+0.5],
        # [ 0.6,  0.5+0.5],
        # [ 2.1,  0.5+0.5],
        # [-0.9, -2.5+0.5],
        # [ 0.6, -2.5+0.5],
        # [ 2.1, -2.5+0.5],

        # [-0.9,  0.5+1.0],
        # [ 0.6,  0.5+1.0],
        # [ 2.1,  0.5+1.0],
        # [-0.9, -2.5+1.0],
        # [ 0.6, -2.5+1.0],
        # [ 2.1, -2.5+1.0],

        # [-0.9,  0.5+1.5],
        # [ 0.6,  0.5+1.5],
        # [ 2.1,  0.5+1.5],
        # [-0.9, -2.5+1.5],
        # [ 0.6, -2.5+1.5],
        # [ 2.1, -2.5+1.5],

        # [-0.9,  0.5+2.0],
        # [ 0.6,  0.5+2.0],
        # [ 2.1,  0.5+2.0],
        # [-0.9, -2.5+2.0],
        # [ 0.6, -2.5+2.0],
        # [ 2.1, -2.5+2.0],



        # [-0.9+0.3,  0.5],
        # [ 0.6+0.3,  0.5],
        # [ 2.1+0.3,  0.5],
        # [-0.9+0.3, -2.5],
        # [ 0.6+0.3, -2.5],
        # [ 2.1+0.3, -2.5],

        # [-0.9+0.3,  0.5+0.5],
        # [ 0.6+0.3,  0.5+0.5],
        # [ 2.1+0.3,  0.5+0.5],
        # [-0.9+0.3, -2.5+0.5],
        # [ 0.6+0.3, -2.5+0.5],
        # [ 2.1+0.3, -2.5+0.5],

        # [-0.9+0.3,  0.5+1.0],
        # [ 0.6+0.3,  0.5+1.0],
        # [ 2.1+0.3,  0.5+1.0],
        # [-0.9+0.3, -2.5+1.0],
        # [ 0.6+0.3, -2.5+1.0],
        # [ 2.1+0.3, -2.5+1.0],

        # [-0.9+0.3,  0.5+1.5],
        # [ 0.6+0.3,  0.5+1.5],
        # [ 2.1+0.3,  0.5+1.5],
        # [-0.9+0.3, -2.5+1.5],
        # [ 0.6+0.3, -2.5+1.5],
        # [ 2.1+0.3, -2.5+1.5],

        # [-0.9+0.3,  0.5+2.0],
        # [ 0.6+0.3,  0.5+2.0],
        # [ 2.1+0.3,  0.5+2.0],
        # [-0.9+0.3, -2.5+2.0],
        # [ 0.6+0.3, -2.5+2.0],
        # [ 2.1+0.3, -2.5+2.0],

        # [-0.9+0.15,  0.5],
        # [ 0.6+0.15,  0.5],
        # [ 2.1+0.15,  0.5],
        # [-0.9+0.15, -2.5],
        # [ 0.6+0.15, -2.5],
        # [ 2.1+0.15, -2.5],

        [-0.9+0.15,  0.5+0.15],
        [ 0.6+0.15,  0.5+0.15],
        [ 2.1+0.15,  0.5+0.15],
        [-0.9+0.15, -2.5+0.15],
        [ 0.6+0.15, -2.5+0.15],
        [ 2.1+0.15, -2.5+0.15],

        [-0.9+0.15,  0.5+0.25],
        [ 0.6+0.15,  0.5+0.25],
        [ 2.1+0.15,  0.5+0.25],
        [-0.9+0.15, -2.5+0.25],
        [ 0.6+0.15, -2.5+0.25],
        [ 2.1+0.15, -2.5+0.25],

        [-0.9+0.15,  0.5+0.5],
        [ 0.6+0.15,  0.5+0.5],
        [ 2.1+0.15,  0.5+0.5],
        [-0.9+0.15, -2.5+0.5],
        [ 0.6+0.15, -2.5+0.5],
        [ 2.1+0.15, -2.5+0.5],

        [-0.9+0.15,  0.5+0.75],
        [ 0.6+0.15,  0.5+0.75],
        [ 2.1+0.15,  0.5+0.75],
        [-0.9+0.15, -2.5+0.75],
        [ 0.6+0.15, -2.5+0.75],
        [ 2.1+0.15, -2.5+0.75],

        [-0.9+0.15,  0.5+1.0],
        [ 0.6+0.15,  0.5+1.0],
        [ 2.1+0.15,  0.5+1.0],
        [-0.9+0.15, -2.5+1.0],
        [ 0.6+0.15, -2.5+1.0],
        [ 2.1+0.15, -2.5+1.0],

        [-0.9+0.15,  0.5+1.25],
        [ 0.6+0.15,  0.5+1.25],
        [ 2.1+0.15,  0.5+1.25],
        [-0.9+0.15, -2.5+1.25],
        [ 0.6+0.15, -2.5+1.25],
        [ 2.1+0.15, -2.5+1.25],

        [-0.9+0.15,  0.5+1.5],
        [ 0.6+0.15,  0.5+1.5],
        [ 2.1+0.15,  0.5+1.5],
        [-0.9+0.15, -2.5+1.5],
        [ 0.6+0.15, -2.5+1.5],
        [ 2.1+0.15, -2.5+1.5],

        [-0.9+0.15,  0.5+1.75],
        [ 0.6+0.15,  0.5+1.75],
        [ 2.1+0.15,  0.5+1.75],
        [-0.9+0.15, -2.5+1.75],
        [ 0.6+0.15, -2.5+1.75],
        [ 2.1+0.15, -2.5+1.75],

        [-0.9+0.15,  0.5+1.85],
        [ 0.6+0.15,  0.5+1.85],
        [ 2.1+0.15,  0.5+1.85],
        [-0.9+0.15, -2.5+1.85],
        [ 0.6+0.15, -2.5+1.85],
        [ 2.1+0.15, -2.5+1.85],

        # [-0.9+0.15,  0.5+2.0],
        # [ 0.6+0.15,  0.5+2.0],
        # [ 2.1+0.15,  0.5+2.0],
        # [-0.9+0.15, -2.5+2.0],
        # [ 0.6+0.15, -2.5+2.0],
        # [ 2.1+0.15, -2.5+2.0],
        ]

    for i in np.arange(0.0, 7.0, 0.5):
        obst_pos.append([-3.5+i, 3.5  ])
        obst_pos.append([ 3.5-i,-3.5  ])
        obst_pos.append([-3.5,  -3.5+i])
        obst_pos.append([ 3.5,   3.5-i])

    manualPlot = ManualPlot(maxLen=4)
    potentialField = PotentialField(obst_pos)

    while True:
        # if manualPlot.click_flag:
        #     adam.reset()

        optimizer.zero_grad()

        z = potentialField.calc(x, manualPlot.tgtPos)
        z.backward()
        nn.utils.clip_grad_norm_([x], 0.5)

        # with torch.no_grad(): 
        #     dx = torch.clamp(x.grad, -1.0, 1.0)
        #     # dx = x.grad
        #     # dx_hat = adam.update(dx)

        #     x = x - eta * dx
        #     # x = x - dx_hat
        #     # print(dx_hat.detach().numpy())
        #     print(dx.numpy())
        optimizer.step()
        # x.requires_grad = True 

        print(x.detach().numpy())

        manualPlot.clear()
        manualPlot.drawMap()
        manualPlot.draw_points(np.array([manualPlot.tgtPos]), psize=10)
        manualPlot.draw_points(x.detach().numpy(), psize=10, color='blue')
        manualPlot.update(0.001)


if __name__ == '__main__':

    test1()