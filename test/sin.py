import numpy as np
import flow as fl
import progressbar as pb
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

def answer(x, error=0.0):
    return np.sin(2 * np.pi * x) + np.random.rand(*(x.shape)) * error
    
train_x  = np.expand_dims(np.arange(-5, 5, 10 / 300), 1)
train_y = answer(train_x)

def test(train=True):

    sess =  fl.Session()
    sess.fan_in = 1
    sess.fan_out = 1

    x = fl.Placeholder(sess, train_x, 'x')
    y = fl.Placeholder(sess, train_y, 'y')

    S = x
    for _ in range(7):
        S, _, _ = fl.fully_conntected(S, 100, activation=fl.tanh, initializer=fl.xavier_initializer())
    S, _, _ = fl.fully_conntected(S, 1, activation=None, initializer=fl.xavier_initializer())

    y_ = S
    # E = fl.avg(fl.avg(fl.square(y - y_), 0), 0)
    E = fl.l2loss(y, y_)

    optimizer = fl.AdamOptimizer(sess, [E], lr=0.001)
    
    if False:  # Pre-training before animation
        for _ in pb.progressbar(range(epoch)):
            train()

    anim = fl.make_animation1d(x, y, y_, E, optimizer, (-4, 4), (-2, 2), answer, interval=1, blit=True)
                            
    plt.show()
