import numpy as np
import flow as fl

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

def read_x(pathname):
    with open(pathname, 'rb') as fd:
        raw = fd.read()
    info = np.frombuffer(raw[:16], dtype='>u4')
    images = np.frombuffer(raw[16:], dtype=np.byte)
    magic_number, num_images, num_rows, num_cols = info
    images = np.reshape(images, (num_images, num_rows, num_cols))
    return num_images, np.float32(images / 255.0), num_rows, num_cols

def read_y(pathname):
    with open(pathname, 'rb') as fd:
        raw = fd.read()
    info = np.frombuffer(raw[:8], dtype='>u4')
    labels = np.frombuffer(raw[8:], dtype=np.byte)
    magic_number, num_items = info

    eyes = np.eye(10)

    y = np.zeros((num_items, 10))
    for i in range(num_items):
        y[i] = eyes[labels[i]]

    return num_items, np.array(y, dtype=np.float32)

def test():

    num_images, images, num_rows, num_cols = read_x('data/mnist/train_x')
    _, labels = read_y('data/mnist/train_y')

    images = np.reshape(images, (num_images, 28 * 28))

    sess = fl.Session()
    sess.fan_in = 28*28
    sess.fan_out = 10

    input_x = fl.Placeholder(sess, images, 'x')
    output_y = fl.Placeholder(sess, labels, 'y')

    S0, W0, b0 = fl.fully_conntected(input_x, 256, activation=None, initializer=fl.xavier_initializer())
    S1, W1, b1 = fl.fully_conntected(S0, 256, activation=None, initializer=fl.xavier_initializer())
    S2, W2, b2 = fl.fully_conntected(S1, 10, activation=None, initializer=fl.xavier_initializer())

    y_ = S2
    E = fl.l2loss(output_y, y_)
    optimizer = fl.AdamOptimizer(sess, lr=0.01)
    
    for _ in range(100):
        optimizer.minimize(E)
        print('E:', E.result)
