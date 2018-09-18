import flow as fl
import numpy as np
import progressbar as pb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

train_x = np.array([
    [1.2, 5.3],
    [1.1, 5.4],
    [1.4, 4.5],
    [1.0, 6.0],
    [5.3, 1.2],
    [5.4, 1.1],
    [4.5, 1.5],
    [6.0, 1.0]
])

train_y = np.expand_dims(np.array([
    0, 0, 0, 0, 1, 1, 1, 1
]), 1)

epoch = 1000

def test():

    sess = fl.Session()
    sess.fan_in = train_x.shape[-1]
    sess.fan_out = train_y.shape[-1]

    x = fl.Placeholder(sess, train_x, 'x')
    y = fl.Placeholder(sess, train_y, 'y')

    S0, W0, b0 = fl.fully_conntected(x, 2, activation=fl.sigmoid, initializer=fl.xavier_initializer())
    S1, W1, b1 = fl.fully_conntected(S0, 1, activation=fl.sigmoid, initializer=fl.xavier_initializer())

    y_ = S1
    E = fl.l2loss(y_, y)
    optimizer = fl.AdamOptimizer(sess, lr=0.01)

    weight_vectors = {
        'W0': W0,
        'b0': b0,
        'W1': W1,
        'b1': b1
    }

    start_error = E.get_result()
    for _ in pb.progressbar(range(epoch)):
        optimizer.minimize(E)
    
    def get_result(q):
        a = np.zeros((8, 2))
        a[0] = np.array(q)
        x.result = a
        return y_.get_result()[0,0]

    print('Starting error:', start_error)
    print('Final error:', E.result)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 7), ylim=(0, 7))
    red_scatter = ax.scatter([], [])
    blue_scatter = ax.scatter([], [])
    train_x_scatter = ax.scatter(train_x[:4,0], train_x[:4,1])
    train_y_scatter = ax.scatter(train_x[4:,0], train_x[4:,1])

    def anim_init():
        red_scatter.set_offsets([])
        blue_scatter.set_offsets([])
        return red_scatter, blue_scatter
    
    def anim_update(i):
        optimizer.minimize(E)

        result_x = []
        result_y = []
        for i in np.arange(0, 7, 0.1):
            for j in np.arange(0, 7, 0.1):
                a = np.array([i, j])
                result_x.append(a)
                result_y.append(get_result(a))
            
        red  = []
        blue = []
        for idx, rex in enumerate(result_x):
            if result_y[idx] > 0.5:
                red.append(rex)
            else:
                blue.append(rex)
        red = np.array(red)
        blue = np.array(blue)

        red_scatter.set_offsets(red)
        blue_scatter.set_offsets(blue)

        return red_scatter, blue_scatter, train_x_scatter, train_y_scatter
    
    anim = animation.FuncAnimation(fig, anim_update, init_func=anim_init,
                                frames=100, interval=80, blit=True)

    plt.show()