import numpy as np
import flow as fl
import progressbar as pb
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle

M = 10
N = 500
lr = 0.000001
epoch = 20000

def answer(x, error=0.0):
    return np.sin(2 * np.pi * x) + np.random.rand(*(x.shape)) * error

def test(train=True):

    sess =  fl.Session(lr=lr)

    def initializer(*shape):
        return np.random.rand(*shape) * 2 - 1.0

    input_size = 1
    h1 = 200
    h2 = 200
    output_size = 2

    W0 = fl.Variable(sess, initializer(input_size, h1))
    b0 = fl.Variable(sess, np.zeros(h1))
    W1 = fl.Variable(sess, initializer(h1, h2))
    b1 = fl.Variable(sess, np.zeros(h2))
    W2 = fl.Variable(sess, initializer(h2, output_size))
    b2 = fl.Variable(sess, np.zeros(output_size))

    weight_vectors = {
        'W0': W0,
        'b0': b0,
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

    x = fl.Placeholder(sess, train_x, 'x')
    y = fl.Placeholder(sess, train_y, 'y')
    
    activation = fl.tanh

    S0 = activation(fl.matmul(x, W0) + b0)
    S1 = activation(fl.matmul(S0, W1) + b1)
    S2 = activation(fl.matmul(S1, W2) + b2)

    p = fl.select(S2, 1, 0, 1)
    q = fl.select(S2, 1, 1, 2)

    y_ = p / q
    E = fl.sum(fl.sum(fl.square(y - y_), 0), 0)

    def save():
        fd = open('save.sav', 'wb')
        np.savez(fd, **weight_vectors)
        fd.close()

    def load():
        try:
            fd = open('save.sav', 'rb')
            data = np.load(fd)
            for key in weight_vectors:
                value = weight_vectors[key]
                value.result = data[key]
            fd.close()
        except Exception as e:
            print('Failed to load')
    
    def train():
        x.result = np.random.rand(500) * 8 - 4
        x.result.sort()
        x.result = np.expand_dims(x.result, 1)
        y.result = answer(x.result)
        E.minimize()

    answer_x = np.arange(-4, 4, 0.01)
    answer_y = answer(answer_x)

    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 4), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)
    ans, = ax.plot(answer_x, answer_y)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        train()
        gx = np.sum(x.result, 1)
        gy = np.sum(y_.result, 1)
        line.set_data(gx, gy)
        print('E:', E.result)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=2000, interval=1, blit=True)

    plt.show()

    save()

    # plt.plot(gx, gy)
    # plt.plot(gx, ans, 'r--')
    # plt.show()
    
