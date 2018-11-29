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

    y = np.zeros((num_items, 10))
    y[np.arange(num_items),labels] = 1

    return num_items, np.array(y, dtype=np.float32)

def mini_batch(x, y, sz):
    n = len(x)
    for i in range(0, n, sz):
        yield x[i:i+sz], y[i:i+sz]

def test():

    num_images, images, num_rows, num_cols = read_x('data/mnist/train_x')
    _, labels = read_y('data/mnist/train_y')
    vec_size = 28 * 28
    batch_size = 6000

    num_test, test_images, _, _ = read_x('data/mnist/test_x')
    _, test_labels = read_y('data/mnist/test_y')

    print(num_images, num_test, 'Images Read')

    images = np.reshape(images, (num_images, vec_size))
    test_images = np.reshape(test_images, (num_test, vec_size))

    sess = fl.Session()
    sess.fan_in = vec_size
    sess.fan_out = 10

    input_x = fl.Placeholder(sess, images, 'x')
    output_y = fl.Placeholder(sess, labels, 'y')

    S0, _, _ = fl.fully_conntected(input_x, 256, activation=fl.gelu, initializer=fl.xavier_initializer())
    S1, _, _ = fl.fully_conntected(S0, 128, activation=fl.gelu, initializer=fl.xavier_initializer())
    S2, _, _ = fl.fully_conntected(S1, 10, activation=fl.gelu, initializer=fl.xavier_initializer())

    y_ = S2
    E = fl.softmax_cross_entropy_loss(output_y, y_, 1)
    # E = fl.l2loss(output_y, y_)
    optimizer = fl.AdamOptimizer(sess, [E], lr=0.001)
    
    for _ in range(10):
        input_x.set_result(test_images)
        output_y.set_result(test_labels)
        print('E:', E.get_result() / num_test)
        print('acc:', np.sum(np.argmax(y_.get_result(), axis=1) == np.argmax(test_labels, 1)) / num_test)
        for _ in range(10):
            for batch_x, batch_y in mini_batch(images, labels, batch_size):
                input_x.set_result(batch_x)
                output_y.set_result(batch_y)
                optimizer.minimize()
            print('E:', E.get_result() / batch_size)
        
