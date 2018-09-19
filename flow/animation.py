import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

def show_animation(x, y, y_, E, optimizer, xlim, ylim, print_error=True):

    train_x = x.get_result()
    train_y = y.get_result()

    dx = (xlim[1] - xlim[0]) / 100
    dy = (ylim[1] - ylim[0]) / 100

    test_x = []
    for i in np.arange(xlim[0], xlim[1], dx):
        for j in np.arange(ylim[0], ylim[1], dy):
            test_x.append([i, j])
    test_x = np.array(test_x)

    fig = plt.figure()
    ax = plt.axes(xlim=xlim, ylim=ylim)
    red_scatter = ax.scatter([], [])
    blue_scatter = ax.scatter([], [])
    train_scatter = ax.scatter(train_x[:,0], train_x[:,1], c=np.squeeze(train_y))
    
    def anim_update(i):
        x.set_result(train_x)
        optimizer.minimize(E)
        if print_error:
            print('E:', E.get_result())

        # qy = y_.get_result()
        # qy = np.where(qy > 0.5, 1.0, 0.0)
        # print('Wrong:', np.sum(np.absolute(qy - train_y), axis=None))
        # print('diff:', np.squeeze(np.absolute(qy - train_y)))
        # print('qy:', qy)
        # print('train_y:', train_y)  

        x.set_result(test_x)
        test_y = y_.get_result().T[0]
            
        red  = []
        blue = []
        for idx, rex in enumerate(test_x):
            if test_y[idx] > 0.5:
                red.append(rex)
            else:
                blue.append(rex)

        if len(red) > 0:
            red_scatter.set_offsets(np.array(red))
        if len(blue) > 0:
            blue_scatter.set_offsets(np.array(blue))

        return blue_scatter, red_scatter, train_scatter
    
    anim = animation.FuncAnimation(fig, anim_update, interval=1, blit=True)

    plt.show()