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

def relu_grad(a):
    return ReluGradNode(a.sess, [a])

def leaky_relu(a, alpha=0.2):
    return LeakyReluNode(a.sess, [a], alpha)

def leaky_relu_grad(a, grad, alpha=0.2):
    return LeakyReluGradNode(a.sess, [a, grad], alpha)

def tanh(a):
    return TanhNode(a.sess, [a])

def softmax(a):
    return SoftmaxNode(a.sess, [a])

def softmax_grad(a, grad):
    return SoftmaxGradNode(a.sess, [a, grad])

def log(a):
    return LogNode(a.sess, [a])

def log_grad(a, grad):
    return LogGradNode(a.sess, [a, grad])

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

def fold(a, axis, num):
    return FoldNode(a.sess, [a], axis, num)

def repeat(a, axis, count):
    return RepeatNode(a.sess, [a], axis, count)

def select(a, key):
    return SelectNode(a.sess, [a], key)

def reduce_shape(a, shape):
    return ReduceShapeNode(a.sess, [a], shape)

def sum(a, axis):
    return SumNode(a.sess, [a], axis)

def expand_dims(a, axis):
    return ExpandDimsNode(a.sess, [a], axis)

def squeeze(a, axis):
    return SqueezeNode(a.sess, [a], axis)

def reshape(a, shape):
    return ReshapeNode(a.sess, [a], shape)

def avg(a, axis):
    return AvgNode(a.sess, [a], axis)

def conv2d(a, b):
    return Conv2DNode(a.sess, [a, b])

def conv2d_grad(grad, b, filter_wh):
    return Conv2DGradientNode(a.sess, [a, b], filter_wh)
