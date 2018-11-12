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

def matmul(a, b, name=None):
    return fl.MatmulNode(a.sess, [a, b], name=name)

def neg(a, name=None):
    return fl.NegNode(a.sess, [a], name=name)

def add(a, b, name=None):
    return fl.AddNode(a.sess, [a, b], name=name)

def sub(a, b, name=None):
    return fl.SubNode(a.sess, [a, b], name=name)

def mul(a, b, name=None):
    return fl.MulNode(a.sess, [a, b], name=name)

def div(a, b, name=None):
    return fl.DivNode(a.sess, [a, b], name=name)

def sigmoid(a, name=None):
    return fl.SigmoidNode(a.sess, [a], name=name)

def relu(a, name=None):
    return fl.ReluNode(a.sess, [a], name=name)

def relu_grad(a, name=None):
    return fl.ReluGradNode(a.sess, [a], name=name)

def leaky_relu(a, alpha=0.2, name=None):
    return fl.LeakyReluNode(a.sess, [a], alpha, name=name)

def leaky_relu_grad(a, grad, alpha=0.2, name=None):
    return fl.LeakyReluGradNode(a.sess, [a, grad], alpha, name=name)

def tanh(a, name=None):
    return fl.TanhNode(a.sess, [a], name=name)

def softmax(a, name=None):
    return fl.SoftmaxNode(a.sess, [a], name=name)

def softmax_grad(a, grad, name=None):
    return fl.SoftmaxGradNode(a.sess, [a, grad], name=name)

def log(a, name=None):
    return fl.LogNode(a.sess, [a], name=name)

def log_grad(a, grad, name=None):
    return fl.LogGradNode(a.sess, [a, grad], name=name)

def exp(a, name=None):
    return fl.ExpNode(a.sess, [a], name=name)

def square(a, name=None):
    return fl.SquareNode(a.sess, [a], name=name)

def l2loss(a, b, name=None):
    return fl.L2LossNode(a.sess, [a, b], name=name)

def transpose(a, name=None):
    return fl.TransposeNode(a.sess, [a], name=name)

def concat(a, b, axis=0, name=None):
    return fl.ConcatenateNode(a.sess, [a, b], axis, name=name)

def fold(a, axis, num, name=None):
    return fl.FoldNode(a.sess, [a], axis, num, name=name)

def repeat(a, axis, count, name=None):
    return fl.RepeatNode(a.sess, [a], axis, count, name=name)

def select(a, key, name=None):
    return fl.SelectNode(a.sess, [a], key, name=name)

def reduce_shape(a, shape, name=None):
    return fl.ReduceShapeNode(a.sess, [a], shape, name=name)

def sum(a, axis, name=None):
    return fl.SumNode(a.sess, [a], axis, name=name)

def expand_dims(a, axis, name=None):
    return fl.ExpandDimsNode(a.sess, [a], axis, name=name)

def squeeze(a, axis, name=None):
    return fl.SqueezeNode(a.sess, [a], axis, name=name)

def reshape(a, shape, name=None):
    return fl.ReshapeNode(a.sess, [a], shape, name=name)

def avg(a, axis, name=None):
    return fl.AvgNode(a.sess, [a], axis, name=name)

def conv2d(a, b, name=None):
    return fl.Conv2DNode(a.sess, [a, b], name=name)

def conv2d_grad(grad, b, filter_wh, name=None):
    return fl.Conv2DGradientNode(a.sess, [a, b], filter_wh, name=name)

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
            