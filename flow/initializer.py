import numpy as np
from flow.node.util import *

def xavier_initializer(uniform=False):
    def initializer(sess, shape):
        return xavier(shape, sess.fan_in, sess.fan_out, uniform)
    
    return initializer

def rand_initializer(min_value=0.0, max_value=1.0):
    diff = max_value - min_value
    def initializer(sess, shape):
        return np.random.rand(*shape) * diff - min_value
        
    return initializer

def randn_initializer(min_value=-1.0, max_value=1.0):
    diff = max_value - min_value
    def initializer(sess, shape):
        return np.random.randn(*shape) * diff - min_value
    
    return initializer

def zero_initializer():
    def initializer(sess, shape):
        return np.zeros(shape)
    
    return initializer
