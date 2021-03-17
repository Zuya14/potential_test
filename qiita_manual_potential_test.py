import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

class PotentialField:

    def __init__(self, pos):
        self.pos = torch.tensor(pos)

    def calc(self, x, g):
        repulsive_force = torch.sum(torch.cat([self.calc_RepulsivePotential(x, p).unsqueeze(0) for p in self.pos]), dim=0)

        attractive_force = self.calc_AttractivePotential(x, g)

        return torch.clamp(attractive_force + repulsive_force, max=10)

    def calc_AttractivePotential(self, x, p, w=1.0, d_thr=1.0):
        d = torch.sqrt(torch.sum((x-p)**2, dim=-1) + 1e-6)

        return torch.where(
            d > d_thr,
            d_thr * w * d - w / 2.0 * d_thr**2,
            w / 2.0 * d**2
            )

    def calc_RepulsivePotential(self, x, p, w=0.1, d_thr=1.0):
        d = torch.sqrt(torch.sum((x-p)**2, dim=-1) + 1e-6)

        return torch.where(
            d > d_thr,
            0.0*d,
            0.5*w * (1.0/d + 1.0/d_thr)**2
            )

class Plot:

    def __init__(self, maxLen=None):

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)

        if maxLen:
            self.ax.set_xlim(-maxLen, maxLen)
            self.ax.set_ylim(-maxLen, maxLen)

        self.ax.set_aspect('equal')
        self.ax.grid()

        self.maxLen = maxLen
        self.update()

    def onKey(self, event):
        if event.key == 'q':
            exit()

    def update(self, interval=0.01):
        plt.pause(interval)

    def clear(self):
        self.ax.lines.clear()
        self.ax.collections.clear()
        self.ax.patches.clear()

    def draw_points(self, points, psize=1, color='red', marker='o', alpha=1):
        p = points.reshape((-1, 2))
        self.ax.scatter(p[:,0], p[:,1], s=psize, c=color, marker=marker, alpha=alpha)

    def draw_heatmap(self, X, Y, data):
        self.ax.pcolormesh(X, Y, data, cmap='cool', vmin=data.min(), vmax=data.max())

    def save(self, name):
        self.fig.savefig(name)

class ManualPlot(Plot):
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

    def getFlag(self):
        return self.click_flag

    def offFlag(self):
        self.click_flag = False

if __name__ == '__main__':

    maxLen = 5.0
    step = maxLen*0.01
    xs = np.arange(-maxLen, maxLen+1e-6, step)
    ys = np.arange(-maxLen, maxLen+1e-6, step)
    X, Y = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[X.flatten(), Y.flatten()], dtype=torch.float)

    x = torch.zeros(2, requires_grad=True)

    optimizer = optim.SGD([x], lr=0.01, momentum=0.8)

    obst_pos = [
        [1.0, 0.0],
        [1.0, 1.5],
    ]

    manualPlot = ManualPlot(maxLen)
    potentialField = PotentialField(obst_pos)

    with torch.no_grad():
        grid_force = potentialField.calc(grid, torch.tensor(manualPlot.tgtPos)).view(X.shape[0], -1).numpy()

    while True:

        optimizer.zero_grad()

        force = potentialField.calc(x, torch.tensor(manualPlot.tgtPos))
        force.backward()

        optimizer.step()

        if manualPlot.getFlag():
            grid_force = potentialField.calc(grid, torch.tensor(manualPlot.tgtPos)).view(X.shape[0], -1).numpy()
            manualPlot.offFlag()

        manualPlot.clear()
        manualPlot.draw_heatmap(X, Y, grid_force)
        manualPlot.draw_points(np.array([manualPlot.tgtPos]), psize=10, color='white')
        manualPlot.draw_points(x.detach().numpy(), psize=10, color='red')
        manualPlot.update(0.01)
