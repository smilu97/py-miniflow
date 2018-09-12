from flow.node import *

def matmul(a, b):
    return MatmulNode(a.sess, [a, b])

def add(a, b):
    return AddNode(a.sess, [a, b])

def sub(a, b):
    return SubNode(a.sess, [a, b])

def sigmoid(a):
    return SigmoidNode(a.sess, [a])

def relu(a):
    return ReluNode(a.sess, [a])

def softmax(a):
    return SoftmaxNode(a.sess, [a])

def log(a):
    return LogNode(a.sess, [a])

def square(a):
    return SquareNode(a.sess, [a])

def transpose(a):
    return TransposeNode(a.sess, [a])

def concat(a, b, axis=0):
    return ConcatenateNode(a.sess, [a, b], axis)

def sum(a, axis):
    return SumNode(a.sess, [a], axis)

def conv2d(a, b):
    return Conv2DNode(a.sess, [a, b])
