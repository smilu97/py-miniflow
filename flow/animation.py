import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

def make_animation1d(x, y, y_, E, optimizer, xlim, ylim, answer, print_error=True, epoch_per_frame=1, **kwargs):

    train_x = np.squeeze(x.get_result())
    train_y = np.squeeze(y.get_result())
    

    test_x = np.expand_dims(np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 300), 1)
    test_y = answer(test_x)

    fig = plt.figure()
    ax = plt.axes(xlim=xlim, ylim=ylim)
    line, = ax.plot([], [], lw=2)
    ans, = ax.plot(test_x, test_y)

    def animate(i):
        for _ in range(epoch_per_frame):
            optimizer.minimize()
        if print_error:
            print('E:', E.get_result())
        line.set_data(train_x, np.squeeze(y_.get_result()))
        return line,

    return animation.FuncAnimation(fig, animate, **kwargs)

def make_animation2d(x, y, y_, E, optimizer, xlim, ylim, print_error=True, epoch_per_frame=1, **kwargs):

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

        x.set_result(test_x)
        test_y = y_.get_result().T[0]

        x.set_result(train_x)
        for _ in range(epoch_per_frame):
            optimizer.minimize()
        if print_error:
            print('E:', E.get_result())
            
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
    
    return animation.FuncAnimation(fig, anim_update, **kwargs)
