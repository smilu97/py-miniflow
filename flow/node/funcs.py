import numpy as np
import flow as fl

def zeros(sess, shape, name='zero'):
    return fl.Placeholder(sess, np.zeros(shape), name)

def ones(sess, shape, name='ones'):
    return fl.Placeholder(sess, np.ones(shape), name)

def empty(sess, shape, name='empty'):
    return fl.Placeholder(sess, np.empty(shape), name)

def zeros_like(sess, a, name='zero'):
    return fl.Placeholder(sess, np.zeros_like(a.shape), name)

def ones_like(sess, a, name='ones'):
    return fl.Placeholder(sess, np.ones_like(a.shape ), name)

def empty_like(sess, a, name='empty'):
    return fl.Placeholder(sess, np.empty_like(a.shape), name)

def matmul(a, b):
    return fl.MatmulNode(a.sess, [a, b])

def neg(a):
    return fl.NegNode(a.sess, [a])

def add(a, b):
    return fl.AddNode(a.sess, [a, b])

def sub(a, b):
    return fl.SubNode(a.sess, [a, b])

def mul(a, b):
    return fl.MulNode(a.sess, [a, b])

def div(a, b):
    return fl.DivNode(a.sess, [a, b])

def sigmoid(a):
    return fl.SigmoidNode(a.sess, [a])

def relu(a):
    return fl.ReluNode(a.sess, [a])

def relu_grad(a):
    return fl.ReluGradNode(a.sess, [a])

def leaky_relu(a, alpha=0.2):
    return fl.LeakyReluNode(a.sess, [a], alpha)

def leaky_relu_grad(a, grad, alpha=0.2):
    return fl.LeakyReluGradNode(a.sess, [a, grad], alpha)

def tanh(a):
    return fl.TanhNode(a.sess, [a])

def softmax(a):
    return fl.SoftmaxNode(a.sess, [a])

def softmax_grad(a, grad):
    return fl.SoftmaxGradNode(a.sess, [a, grad])

def log(a):
    return fl.LogNode(a.sess, [a])

def log_grad(a, grad):
    return fl.LogGradNode(a.sess, [a, grad])

def exp(a):
    return fl.ExpNode(a.sess, [a])

def square(a):
    return fl.SquareNode(a.sess, [a])

def l2loss(a, b):
    return fl.L2LossNode(a.sess, [a, b])

def transpose(a):
    return fl.TransposeNode(a.sess, [a])

def concat(a, b, axis=0):
    return fl.ConcatenateNode(a.sess, [a, b], axis)

def fold(a, axis, num):
    return fl.FoldNode(a.sess, [a], axis, num)

def repeat(a, axis, count):
    return fl.RepeatNode(a.sess, [a], axis, count)

def select(a, key):
    return fl.SelectNode(a.sess, [a], key)

def reduce_shape(a, shape):
    return fl.ReduceShapeNode(a.sess, [a], shape)

def sum(a, axis):
    return fl.SumNode(a.sess, [a], axis)

def expand_dims(a, axis):
    return fl.ExpandDimsNode(a.sess, [a], axis)

def squeeze(a, axis):
    return fl.SqueezeNode(a.sess, [a], axis)

def reshape(a, shape):
    return fl.ReshapeNode(a.sess, [a], shape)

def avg(a, axis):
    return fl.AvgNode(a.sess, [a], axis)

def conv2d(a, b):
    return fl.Conv2DNode(a.sess, [a, b])

def conv2d_grad(grad, b, filter_wh):
    return fl.Conv2DGradientNode(a.sess, [a, b], filter_wh)

def gradients(ys, xs):

    def clean_parent_num(x):
        x.grad_parent_num = 0
        for y in x.children:
            clean_parent_num(y)
    for y in ys: clean_parent_num(y)

    def clean_back_grad(x):
        x.grad_cache = None
        x.grad_recv_num = 0
        for y in x.children:
            if y.grad_parent_num == 0:
                clean_back_grad(y)
            y.grad_parent_num += 1
        
    for y in ys: clean_back_grad(y)
    for y in ys: y.grad_cache = ones(y.sess, y.shape)

    def calc_back_grad(x):
        xc = x.__class__
        if len(x.children) == 0:
            return
        if not hasattr(xc, 'calc_gradients'):
            raise Exception('Cannot calc grad from {}'.format(x.name))
        grads = xc.calc_gradients(x, x.grad_cache)
        for idx, child in enumerate(x.children):
            if child.grad_cache is None:
                child.grad_cache = grads[idx]
            else:
                child.grad_cache += grads[idx]
            child.grad_recv_num += 1
            if child.grad_recv_num >= child.grad_parent_num:
                calc_back_grad(child)

    for y in ys: calc_back_grad(y)
    res = [x.grad_cache if hasattr(x, 'grad_cache') else None for x in xs]
    for y in ys: clean_back_grad(y)

    return res
            