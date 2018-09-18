from flow.node.funcs import *
from flow.initializer import *

def fully_conntected(x, output_size, activation=sigmoid, initializer=zero_initializer(), bias_initializer=zero_initializer()):
    sess = x.sess

    W_init = initializer(sess, (x.shape[-1], output_size))
    W = Variable(x.sess, W_init)
    
    b_init = bias_initializer(sess, (output_size,))
    b = Variable(x.sess, b_init)

    R = matmul(x, W) + b
    if activation is not None:
        R = activation(R)

    return R, W, b