import numpy as np
import flow as fl
import progressbar as pb
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle

lr = 0.001
epoch = 1000

def answer(x, error=0.0):
    return np.sin(2 * np.pi * x) + np.random.rand(*(x.shape)) * error

def test(train=True):

    sess =  fl.Session()
    
    input_size = 1 # Constant
    h1 = 1000
    h2 = 1000
    output_size = 1 # Constant
    batch_size = 200

    sess.fan_in = input_size
    sess.fan_out = output_size

    x = fl.Placeholder(sess, np.zeros((batch_size, 1)), 'x')
    y = fl.Placeholder(sess, np.zeros((batch_size, 1)), 'y')

    S0, W0, b0 = fl.fully_conntected(x, h1, activation=fl.tanh, initializer=fl.xavier_initializer())
    S1, W1, b1 = fl.fully_conntected(S0, h2, activation=fl.tanh, initializer=fl.xavier_initializer())
    S2, W2, b2 = fl.fully_conntected(S1, output_size, activation=None, initializer=fl.xavier_initializer())

    # p = fl.select(S2, 1, 0, 1)
    # q = fl.select(S2, 1, 1, 2)

    y_ = S2
    E = fl.avg(fl.avg(fl.square(y - y_), 0), 0)

    optimizer = fl.AdamOptimizer(sess, lr=lr)

    weight_vectors = {
        'W0': W0,
        'b0': b0,
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

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
        x.result = np.random.rand(batch_size) * 16 - 8
        x.result.sort()
        x.result = np.expand_dims(x.result, 1)
        y.result = answer(x.result)
        optimizer.minimize(E)
    
    if False:
        for _ in pb.progressbar(range(epoch)):
            train()

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
                                frames=np.arange(0, 200), interval=3, blit=True)
                            
    plt.show()
    # anim.save('static/sin.gif', dpi=80, writer='imagemagick')

    save()
