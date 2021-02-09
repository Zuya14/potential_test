import numpy as np
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

import math

class PotentialField:

    def __init__(self, pos):
        self.pos = pos

    def calc(self, x, y, gx, gy, max=100.0, min=-100.0):
        temp = []

        for p in self.pos:
            # temp.append(self.calc_cost(x, y, p).unsqueeze(0))
            temp.append(self.calc_RepulsivePotential(x, y, p, w=1.0, d_thr=0.3).unsqueeze(0))

        # temp.append(self.calc_cost(x, y, [-1.5, 0.0], w=-1.0).unsqueeze(0))
        temp.append(self.calc_AttractivePotential(x, y, [-1.5, 0.0], w=1.0, d_thr=1.0).unsqueeze(0))


        print(torch.cat(temp, dim=0).size())

        return torch.clamp(torch.sum(torch.cat(temp), dim=0), max=max, min=min)

    def calc_cost(self, x, y, p, w=1.0):
        return w * torch.reciprocal(torch.norm(torch.tensor([(x - p[0])**2 , (y - p[1])**2])+1e-2, dim=0))
        # return w * torch.reciprocal(torch.norm(torch.tensor([(x - p[0])**2 , (y - p[1])**2])+5e-2, dim=0))

    def calc_AttractivePotential(self, x, y, p, w=1.0, d_thr=1.0):
        dx2 = (x - p[0])**2
        dy2 = (y - p[1])**2
        d = np.sqrt(dx2 + dy2)

        result = np.where(d > d_thr, 
            d_thr * w * d - w / 2.0 * d_thr**2, 
            w / 2.0 * d**2
            )

        return torch.tensor(result)

    def calc_RepulsivePotential(self, x, y, p, w=1.0, d_thr=1.0):
        dx2 = (x - p[0])**2
        dy2 = (y - p[1])**2
        d = np.sqrt(dx2 + dy2)

        result = np.where(d > d_thr, 0.0, w / 2.0 * (1.0/d + (1.0/d_thr))**2)

        # if d > d_thr:
        #     return torch.zeros(2)
        # else:
        #     # return w * torch.reciprocal(torch.norm(torch.tensor([(x - p[0])**2 , (y - p[1])**2])+1e-2, dim=0))
        #     return w * torch.reciprocal(torch.norm(torch.tensor([dx , (y - p[1])**2])+1e-2, dim=0))
        #     return torch.tensor(w / 2.0 * (1.0/d + (1.0/d_thr))**2)
        return torch.tensor(result)

if __name__ == '__main__':

    obst_pos = [
        # [-3.5, -3.5],
        # [ 3.5, -3.5],
        # [-3.5,  3.5],
        # [ 3.5,  3.5],

        # # [-0.9,  0.5],
        # # [ 0.6,  0.5],
        # # [ 2.1,  0.5],
        # # [-0.9, -2.5],
        # # [ 0.6, -2.5],
        # # [ 2.1, -2.5],

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

        # # [-0.9+0.15,  0.5+2.0],
        # # [ 0.6+0.15,  0.5+2.0],
        # # [ 2.1+0.15,  0.5+2.0],
        # # [-0.9+0.15, -2.5+2.0],
        # # [ 0.6+0.15, -2.5+2.0],
        # # [ 2.1+0.15, -2.5+2.0],


        [-0.9,  0.5],
        [ 0.6,  0.5],
        [ 2.1,  0.5],
        [-0.9, -2.5],
        [ 0.6, -2.5],
        [ 2.1, -2.5],

        [-0.9,  0.5+0.5],
        [ 0.6,  0.5+0.5],
        [ 2.1,  0.5+0.5],
        [-0.9, -2.5+0.5],
        [ 0.6, -2.5+0.5],
        [ 2.1, -2.5+0.5],

        [-0.9,  0.5+1.0],
        [ 0.6,  0.5+1.0],
        [ 2.1,  0.5+1.0],
        [-0.9, -2.5+1.0],
        [ 0.6, -2.5+1.0],
        [ 2.1, -2.5+1.0],

        [-0.9,  0.5+1.5],
        [ 0.6,  0.5+1.5],
        [ 2.1,  0.5+1.5],
        [-0.9, -2.5+1.5],
        [ 0.6, -2.5+1.5],
        [ 2.1, -2.5+1.5],

        [-0.9,  0.5+2.0],
        [ 0.6,  0.5+2.0],
        [ 2.1,  0.5+2.0],
        [-0.9, -2.5+2.0],
        [ 0.6, -2.5+2.0],
        [ 2.1, -2.5+2.0],



        [-0.9+0.3,  0.5],
        [ 0.6+0.3,  0.5],
        [ 2.1+0.3,  0.5],
        [-0.9+0.3, -2.5],
        [ 0.6+0.3, -2.5],
        [ 2.1+0.3, -2.5],

        [-0.9+0.3,  0.5+0.5],
        [ 0.6+0.3,  0.5+0.5],
        [ 2.1+0.3,  0.5+0.5],
        [-0.9+0.3, -2.5+0.5],
        [ 0.6+0.3, -2.5+0.5],
        [ 2.1+0.3, -2.5+0.5],

        [-0.9+0.3,  0.5+1.0],
        [ 0.6+0.3,  0.5+1.0],
        [ 2.1+0.3,  0.5+1.0],
        [-0.9+0.3, -2.5+1.0],
        [ 0.6+0.3, -2.5+1.0],
        [ 2.1+0.3, -2.5+1.0],

        [-0.9+0.3,  0.5+1.5],
        [ 0.6+0.3,  0.5+1.5],
        [ 2.1+0.3,  0.5+1.5],
        [-0.9+0.3, -2.5+1.5],
        [ 0.6+0.3, -2.5+1.5],
        [ 2.1+0.3, -2.5+1.5],

        [-0.9+0.3,  0.5+2.0],
        [ 0.6+0.3,  0.5+2.0],
        [ 2.1+0.3,  0.5+2.0],
        [-0.9+0.3, -2.5+2.0],
        [ 0.6+0.3, -2.5+2.0],
        [ 2.1+0.3, -2.5+2.0],
        ]

    for i in np.arange(0.0, 7.0, 0.5):
    # for i in np.arange(0.0, 7.0, 0.25):
        obst_pos.append([-3.5+i, 3.5  ])
        obst_pos.append([ 3.5-i,-3.5  ])
        obst_pos.append([-3.5,  -3.5+i])
        obst_pos.append([ 3.5,   3.5-i])

    
    obst_pos = np.array(obst_pos)

    potentialField = PotentialField(obst_pos)

    x = np.arange(-4, 4, 0.05) 
    y = np.arange(-4, 4, 0.05)  
    x, y = np.meshgrid(x, y)

    z = potentialField.calc(x, y, 1.5, 0.0).numpy()

    fig = plt.figure()
    ax = Axes3D(fig)

    my_cmap = plt.get_cmap('jet')
    ax.scatter3D(x.reshape(-1), y.reshape(-1), z.reshape(-1), marker=".", c = (z).reshape(-1), cmap = my_cmap)

    plt.show()