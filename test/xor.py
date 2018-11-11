import flow as fl
import numpy as np
import progressbar as pb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def test():

    sess = fl.Session()

    # Vx + b = y

    train_x = np.array([[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])
    train_y = np.array([[0],[1],[1],[0]])

    x = fl.Placeholder(sess, train_x, 'x')
    y = fl.Placeholder(sess, train_y, 'y')
    
    def initializer(*shape):
        return fl.xavier(shape, 2, 2)

    V0 = fl.Variable(sess, initializer(2,2))
    b0 = fl.Variable(sess, np.zeros(2))
    S0 = fl.sigmoid(fl.matmul(x, V0) + b0)

    V1 = fl.Variable(sess, initializer(2,1))
    b1 = fl.Variable(sess, np.zeros(1))
    S1 = fl.sigmoid(fl.matmul(S0, V1) + b1)

    y_ = S1
    E = fl.sum(fl.square(y_ - y), axis=0)

    optimizer = fl.AdamOptimizer(sess, lr=0.1)

    if False:  # Pre-calculate before animation
        print('start error:', E.get_result())

        epoch = 10000
        with pb.ProgressBar(max_value=epoch) as bar:
            for i in range(epoch):
                optimizer.minimize(E)
                bar.update(i)
        
        print('last error:', E.get_result())

    anim = fl.make_animation2d(x, y, y_, E, optimizer, (-1, 2), (-1, 2), epoch_per_frame=50, frames=50, interval=80, blit=True)

    if True :
        plt.show()
    else:
        anim.save('static/xor.gif', writer='imagemagick')
        