import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

class Plot:

    def __init__(self, maxLen=None):

        self.fig, self.ax = plt.subplots()

        if maxLen:
            self.ax.set_xlim(-maxLen, maxLen)
            self.ax.set_ylim(-maxLen, maxLen)

        self.ax.set_aspect('equal')
        self.ax.grid()

        self.maxLen = maxLen
        self.update()
    
    def update(self, interval=0.01):
        plt.pause(interval)

    def clear(self):
        self.ax.lines.clear()
        self.ax.collections.clear()
        self.ax.patches.clear()

    def draw_points(self, points, psize=1, color='red', marker='o', alpha=1):
        p = points.reshape((-1, 2))
        self.ax.scatter(p[:,0], p[:,1], s=psize, c=color, marker=marker, alpha=alpha)

    def draw_circles(self, points, psize=1, color='red', marker='o', alpha=1):
        ppi=72
        ax_length = self.ax.bbox.get_points()[1][0] - self.ax.bbox.get_points()[0][0]
        ax_point = ax_length*ppi / self.fig.dpi

        xsize = self.maxLen*2
        fact = ax_point / xsize

        # scatterのマーカーサイズは直径のポイントの二乗を描くため、実スケールの半径をポイントに変換し直径にしておく
        psize *= 2*fact
        
        p = points.reshape((-1, 2))
        self.ax.scatter(p[:,0], p[:,1], s=psize**2, c='none', edgecolors=color, marker=marker, alpha=alpha)

    def draw_lines(self, points, psize=1, color='red', linestyle='solid', alpha=1):
        p = points.reshape((-1, 2))
        self.ax.plot(p[:,0], p[:,1], ms=psize, c=color, marker="o", linestyle=linestyle, alpha=alpha)

    def drawMap(self, lineSegments, color='green', linewidth=1):
        for points in lineSegments:
            s = points[0]
            e = points[1]
            self.ax.plot([s[0], e[0]],[s[1], e[1]], c=color, linewidth=linewidth)

    def save(self, name):
        self.fig.savefig(name)

class EnvPlot(Plot):

    def __init__(self, maxLen=None):
        super().__init__(maxLen)

    def drawMap(self):
        self.drawRoom()

        origins = [
            (-0.9,  0.5),
            ( 0.6,  0.5),
            ( 2.1,  0.5),
            (-0.9, -2.5),
            ( 0.6, -2.5),
            ( 2.1, -2.5)
            ]

        for origin in origins:
            self.drawShelf(origin)

    def drawShelf(self, origin):
        rect = patches.Rectangle(xy=origin, width=0.3, height=2, ec='#000000', fill=False)
        self.ax.add_patch(rect)

    def drawRoom(self):
        rect = patches.Rectangle(xy=(-3.5, -3.5), width=7, height=7, ec='#000000', fill=False)
        self.ax.add_patch(rect)

    def drawRobot(self, point, theta):
        self.draw_circles(np.array([point]), 0.25, color='blue')

        point2 = [point[0] - 0.25*np.sin(theta), point[1] + 0.25*np.cos(theta)]
        self.draw_lines(np.array([point, point2]), 0.25, color='blue')

    def drawAction(self, point, v, theta, w):
        self.ax.quiver(point[0], point[1], v*np.cos(theta), v*np.sin(theta), angles='xy',scale_units='xy', color='red', scale=1)
        self.drawW(1, point, w)

    def drawW(self, radius, point, w):
        if w < 0:
            theta1 = 1+w
            theta2 = 1
            div = -9
        else:
            theta1 = 0
            theta2 = w
            div = 9
        
        color = 'green'
        arc = patches.Arc(
            point,
            radius,
            radius,
            angle=0,
            theta1=theta1*360,
            theta2=theta2*360,
            capstyle='round',
            linestyle='-',
            # lw=1,
            color=color)
        self.ax.add_patch(arc)

        rad = 2.0*w*math.pi
        endX= point[0] + (radius/2)*np.cos(rad)
        endY= point[1] + (radius/2)*np.sin(rad)

        self.ax.add_patch(
            patches.RegularPolygon(
                (endX, endY),         
                3,                       # number of vertices
                radius/div,
                rad, 
                color=color
            )
        )
        # self.ax.set_xlim([point[0]-radius,point[1]+radius]) and self.ax.set_ylim([point[0]-radius,point[1]+radius]) 
        # Make sure you keep the axes scaled or else arrow will distort

if __name__ == '__main__':
    import math

    envPlot = EnvPlot(maxLen=4)

    for t in range(0, 360, 5):
        theta = t * math.pi / 180.0

        envPlot.clear()
        envPlot.drawMap()
        envPlot.drawRobot([-3, 2], theta)
        envPlot.drawAction([-3, 2], 0.7+np.random.rand()/3, theta+np.random.rand()*0.5, 2.0*np.random.rand()-1.0)
        envPlot.update(0.5)