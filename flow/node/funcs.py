from flow.node import *

def matmul(a, b):
    return MatmulNode(a.sess, [a, b])

def neg(a):
    return NegNode(a.sess, [a])

def add(a, b):
    return AddNode(a.sess, [a, b])

def sub(a, b):
    return SubNode(a.sess, [a, b])

def mul(a, b):
    return MulNode(a.sess, [a, b])

def div(a, b):
    return DivNode(a.sess, [a, b])

def sigmoid(a):
    return SigmoidNode(a.sess, [a])

def relu(a):
    return ReluNode(a.sess, [a])

def leaky_relu(a, alpha=0.2):
    return LeakyReluNode(a.sess, [a], alpha)

def tanh(a):
    return TanhNode(a.sess, [a])

def softmax(a):
    return SoftmaxNode(a.sess, [a])

def log(a):
    return LogNode(a.sess, [a])

def exp(a):
    return ExpNode(a.sess, [a])

def square(a):
    return SquareNode(a.sess, [a])

def l2loss(a, b):
    return L2LossNode(a.sess, [a, b])

def transpose(a):
    return TransposeNode(a.sess, [a])

def concat(a, b, axis=0):
    return ConcatenateNode(a.sess, [a, b], axis)

def select(a, axis, begin, end):
    return SelectNode(a.sess, [a], axis, begin, end)

def sum(a, axis):
    return SumNode(a.sess, [a], axis)

def avg(a, axis):
    return AvgNode(a.sess, [a], axis)

def conv2d(a, b):
    return Conv2DNode(a.sess, [a, b])
