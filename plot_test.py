from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import numpy as np
import torch

def func0(x, y):
    temp = torch.tensor([(x - 2.1)**2 , (y + 7.9)**2])
    return torch.norm(temp, dim=0)

def func1(x, y):
    temp = torch.tensor([(x + 10)**2 , (y -1.4)**2])
    return -0.7 * torch.norm(temp, dim=0)

x = np.arange(-100, 100, 5) 
y = np.arange(-100, 100, 5)  
x, y = np.meshgrid(x, y)


z = func0(x, y).numpy() + func1(x, y).numpy()


fig = plt.figure()
ax = Axes3D(fig)

ax.plot(x.reshape(-1), y.reshape(-1), z.reshape(-1), marker=".",linestyle='None')

plt.show()