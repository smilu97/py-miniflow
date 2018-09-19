import flow as fl
import numpy as np
import progressbar as pb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

def answer(a0):
    res = []
    for a in a0:
        d1 = np.sqrt(np.sum(np.square(a - np.array([3, 3])), axis=None))
        if d1 < 2.0:
            res.append(0.0)
        else:
            res.append(1.0)
    return np.expand_dims(np.array(res), 1)

train_size = 300
train_x = np.random.rand(train_size, 2) * 7
train_y = answer(train_x)

test_x = []
for i in np.arange(0, 7, 0.1):
    for j in np.arange(0, 7, 0.1):
        test_x.append([i, j])
test_x = np.array(test_x)

epoch = 1000

def test():

    sess = fl.Session()
    sess.fan_in = 3
    sess.fan_out = 1

    x = fl.Placeholder(sess, train_x, 'x')
    y = fl.Placeholder(sess, train_y, 'y')

    # We can choose to apply kernel to x_input or not
    x2 = fl.concat(x, fl.square(x), 1)
    # x2 = x

    S0, W0, b0 = fl.fully_conntected(x2, 30, activation=fl.sigmoid, initializer=fl.xavier_initializer())
    S1, W1, b1 = fl.fully_conntected(S0, 1, activation=fl.sigmoid, initializer=fl.xavier_initializer())

    y_ = S1
    E = fl.l2loss(y, y_)
    optimizer = fl.AdamOptimizer(sess, lr=0.01)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 7), ylim=(0, 7))
    red_scatter = ax.scatter([], [])
    blue_scatter = ax.scatter([], [])
    train_scatter = ax.scatter(train_x[:,0], train_x[:,1], c=np.squeeze(train_y))
    
    def anim_update(i):
        x.set_result(train_x)
        optimizer.minimize(E)
        print('E:', E.get_result())

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
