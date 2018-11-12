import numpy as np
from numpy.lib.stride_tricks import as_strided
from flow.node.util import *
from flow.node.funcs import *

class Node:

    def __init__(self, sess, children, trainable=False):
        self.sess = sess
        self.children = children
        self.parents = []
        self.parentNum = 0
        if not hasattr(self, 'result'):
            self.result = None
        self.gradient = None
        self.numGradient = 0
        self.trainable = trainable
        self.initializer_props = None

        for child in children:
            child.parentNum += 1
            child.parents.append(self)
        
        sess.register_node(self)

        self.shape = self.calc_shape(*[child.shape for child in children])
    
    def get_name(self):
        if (not hasattr(self, 'name')) or self.name is None:
            self.name = self.calc_name(*[child.get_name() for child in self.children])
        return self.name

    def get_result(self):
        self.result = self.calc_result(*self.get_children_result())
        return self.result
    
    def set_result(self, value):
        self.result = value
        for parent in self.parents:
            parent.set_result(None)
    
    def get_children_result(self):
        return (child.get_result() for child in self.children)
    
    def propagate_gradient(self):
        gradients = self.calc_gradients()
        for idx, child in enumerate(self.children):
            child.add_gradient(gradients[idx])
            child.numGradient += 1
            if child.numGradient >= child.parentNum:
                child.propagate_gradient()
    
    def add_gradient(self, gradient):
        if self.result.shape != gradient.shape:
            for idx, s in enumerate(gradient.shape):
                r = self.result.shape[idx]
                if r != s and r == 1:
                    gradient = np.sum(gradient, axis=idx)
        if self.gradient is None:
            self.gradient = gradient
        else:
            self.gradient += gradient
    
    def calc_result(self):
        return self.result
    
    def calc_gradients(self):
        return []
    
    def calc_shape(self):
        return None
    
    def check_transform_constant(self, x):
        if type(x) == (type(0) or type(0.0)):
            x = Placeholder(self.sess, np.array([x]), str(x))
        return x
    
    def __add__(self, a):
        a = self.check_transform_constant(a)
        return AddNode(self.sess, [self, a])

    def __sub__(self, a):
        a = self.check_transform_constant(a)
        return SubNode(self.sess, [self, a])
    
    def __mul__(self, a):
        a = self.check_transform_constant(a)
        return MulNode(self.sess, [self, a])
    
    def __matmul__(self, a):
        return MatmulNode(self.sess, [self, a])
    
    def __truediv__(self, a):
        a = self.check_transform_constant(a)
        return DivNode(self.sess, [self, a])
    
    def __neg__(self):
        return NegNode(self.sess, [self])
    
    def __getitem__(self, key):
        return SelectNode(self.sess, [self], key)

class Variable(Node):

    def __init__(self, sess, value, **kwargs):
        self.result = np.float32(value)
        super().__init__(sess, [], trainable=True, **kwargs)
    
    def calc_shape(self):
        return self.result.shape
    
    def calc_name(self):
        return 'Var({})'.format(self.result.shape)

class Placeholder(Node):

    def __init__(self, sess, value, name):
        self.result = np.float32(value)
        self.name = name
        sess.register_placeholder(self)
        super().__init__(sess, [])
    
    def calc_shape(self):
        return self.result.shape
        
class MatmulNode(Node):

    def calc_result(self, a, b):
        return np.matmul(a, b)

    @staticmethod
    def calc_gradients(op, grad):
        v0, v1 = op.children
        v0 = fl.transpose(v0)
        v1 = fl.transpose(v1)
        return [grad @ v1, v0 @ grad]
    
    def calc_shape(self, a, b):
        if len(a) != 2 or len(b) != 2:
            raise Exception('Child of matmul should be 2-dimensional array')
        return (a[0], b[1])
    
    def calc_name(self, a, b):
        return 'Matmul({},{})'.format(a, b)

class NegNode(Node):
    
    def calc_result(self, a):
        return -a
    
    @staticmethod
    def calc_gradients(op, grad):
        return [-grad]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Neg({})'.format(a)

class AddNode(Node):

    def calc_result(self, a, b):
        return a + b

    @staticmethod
    def calc_gradients(op, grad):
        return [
            fl.reduce_shape(grad, op.children[0].shape),
            fl.reduce_shape(grad, op.children[1].shape)
        ]
    
    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
    
    def calc_name(self, a, b):
        return 'Add({},{})'.format(a, b)

class SubNode(Node):

    def calc_result(self, a, b):
        return a - b

    @staticmethod
    def calc_gradients(op, grad):
        return [
            fl.reduce_shape(grad, op.children[0].shape),
            -fl.reduce_shape(grad, op.children[1].shape)
        ]
    
    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
    
    def calc_name(self, a, b):
        return 'Sub({},{})'.format(a, b)

class MulNode(Node):

    def calc_result(self, a, b):
        return a * b
    
    @staticmethod
    def calc_gradients(op, grad):
        v0, v1 = children
        return [
            fl.reduce_shape(grad * v1, v0.shape),
            fl.reduce_shape(grad * v0, v1.shape)
        ]

    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
            
    def calc_name(self, a, b):
        return 'Mul({},{})'.format(a, b)

class DivNode(Node):

    def calc_result(self, a, b):
        return a / b
    
    @staticmethod
    def calc_gradients(op, grad):
        v0, v1 = op.children
        return [
            fl.reduce_shape(grad / v1, v0.shape),
            fl.reduce_shape(-v0 / fl.square(v1) * grad, v0.shape)
        ]
    
    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
    
    def calc_name(self, a, b):
        return 'Div({},{})'.format(a, b)

class SigmoidNode(Node):

    def calc_result(self, a):
        return 1.0 / (1 + np.exp(-a))

    @staticmethod
    def calc_gradients(op, grad):
        r = fl.sigmoid(op.children[0])
        return [r * (1 - r) * grad]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Sigmoid({})'.format(a)

class TanhNode(Node):

    def calc_result(self, a):
        return np.tanh(a)
    
    @staticmethod
    def calc_gradients(op, grad):
        r = fl.tanh(op.children[0])
        return [(1 - r) * (1 + r) * grad]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Tanh({})'.format(a)
    
class ReluNode(Node):

    def calc_result(self, a):
        return np.maximum(a, 0)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.relu_grad(op.children[0], grad)]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Relu({})'.format(a)
    
class ReluGradNode(Node):

    def calc_result(self, a, grad):
        return np.heaviside(a, 0) * grad
    
    def calc_shape(self, a, grad):
        return a
    
    def calc_name(self, a, grad):
        return 'ReluGrad({},{})'.format(a, grad)

class LeakyReluNode(Node):

    def __init__(self, sess, children, alpha):
        self.alpha = alpha
        super().__init__(sess, children)

    @staticmethod
    def calc_result(op, grad):
        return [fl.leaky_relu_grad(op.children[0], grad, op.alpha)]
    
    def calc_gradients(self):
        v0 = self.children[0].get_result()
        return [np.where(v0 > 0, 1, self.alpha) * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'LRelu({})({})'.format(self.alpha, a)
    
class LeakyReluGradNode(Node):

    def __init__(self, sess, children, alpha, **kwargs):
        self.alpha = alpha
        super().__init__(sess, children, **kwargs)

    def calc_result(self, a, grad):
        return [np.where(a > 0, 1, self.alpha) * grad]
    
    def calc_shape(self, a, grad):
        return a
    
    def calc_name(self, a, grad):
        return 'LReluGrad({})({},{})'.format(self.alpha, a, grad)

class SoftmaxNode(Node):

    def calc_result(self, a):
        self.exps = np.exp(a - np.max(a))
        return self.exps / np.sum(self.exps)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.softmax_grad(op.children[0], grad)]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Softmax({})'.format(a)

class SoftmaxGradNode(Node):

    def calc_result(self, a, grad):
        exps = np.exp(a - np.max(a))
        expsum = np.sum(exps)
        A = expsum - exps
        return grad * (exps * A + (exps ** 2)) / (A ** 2)
    
    def calc_shape(self, a, grad):
        return a
    
    def calc_name(self, a, grad):
        return 'SoftmaxGradNode({},{})'.format(a, grad)
    
class LogNode(Node):

    def calc_result(self, a):
        return np.log(a)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.log_grad(op.children[0], grad)]
    
    def calc_shape(self, a):
        return a

    def calc_name(self, a):
        return 'Log({})'.format(a)

class LogGradNode(Node):

    def calc_result(self, a, grad):
        return grad / a
    
    def calc_shape(self, a, grad):
        return a
    
    def calc_name(self, a, grad):
        return 'LogGrad({},{})'.format(a, grad)

class ExpNode(Node):

    def calc_result(self, a):
        return np.exp(a)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.exp(op.children[0]) * grad]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Exp({})'.format(a)
    
class SquareNode(Node):

    def calc_result(self, a):
        return a * a

    @staticmethod
    def calc_gradients(op, grad):
        return [op.children[0] * grad * 2]

    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Square({})'.format(a)

class L2LossNode(Node):

    def calc_result(self, a, b):
        self.diff = a - b
        return np.sum(np.square(self.diff) / 2, axis=None)
    
    @staticmethod
    def calc_gradients(op, grad):
        v0, v1 = op.children
        g = (v0 - v1) * grad
        return [
            fl.reduce_shape(g, v0.shape),
            fl.reduce_shape(-g, v1.shape)
        ]

    def calc_shape(self, a, b):
        return (1,)    

    def calc_name(self, a, b):
        return 'L2Loss({},{})'.format(a, b)

class ReduceShapeNode(Node):

    def __init__(self, sess, children, shape, **kwargs):
        self.shape = shape
        super().__init__(sess, children, **kwargs)

    def calc_result(self, a):
        return array_fit_to_shape(a, self.shape)

    def calc_shape(self, a):
        return self.shape
    
    def calc_name(self, a):
        return a

class SumNode(Node):

    def __init__(self, sess, children, axis, **kwargs):
        self.axis = axis
        self.num = children[0].shape[axis]
        super().__init__(sess, children, **kwargs)

    def calc_result(self, a):
        return np.sum(a, axis=self.axis)

    @staticmethod
    def calc_gradients(op, grad):
        return [fl.repeat(fl.expand_dims(grad, op.axis), op.axis, op.num)]
    
    def calc_shape(self, a):
        res = list(a)
        x = self.axis
        res = res[:x] + res[x+1:]
        return tuple(res)
    
    def calc_name(self, a):
        return 'Sum({})'.format(a)

class RepeatNode(Node):

    def __init__(self, sess, children, axis, count, **kwargs):
        self.axis = axis
        self.count = count
        super().__init__(sess, children, **kwargs)
    
    def calc_result(self, a):
        return np.repeat(a, self.count, axis=self.axis)
    
    @staticmethod
    def calc_gradients(op, grad):
        return fl.fold(grad, op.axis, grad.shape[op.axis] / op.count)
    
    def calc_shape(self, a):
        res = list(a)
        x = self.axis
        res[x] *= self.count
        return tuple(res)
    
    def calc_name(self, a, count, axis):
        return 'Repeat({},{},{})'.format(a, count, axis)
    
class FoldNode(Node):

    def __init__(self, sess, children, axis, num, **kwargs):
        if children[0].shape[axis] % num != 0:
            raise Exception('Not foldable')
        
        self.axis = axis
        self.num = num
        self.fold_cnt = children[0].shape[axis] / num
    
    def calc_result(self, a):
        axis = self.axis
        num = self.num
        fold_cnt = self.fold_cnt
        
        m_shape = list(a.shape)
        m_shape[axis] = num

        m = as_strided(a, tuple(m_shape) + (fold_cnt,), a.strides + (a.strides[axis],))
        return np.sum(m, axis=-1)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.repeat(grad, op.axis, op.fold_cnt)]
    
    def calc_shape(self, a):
        res = list(a)
        res[self.axis] = self.num
        return tuple(res)
    
    def calc_name(self, a):
        return 'Fold({})'.format(a)


class ExpandDimsNode(Node):

    def __init__(self, sess, children, axis, **kwargs):
        self.axis = axis
        super().__init__(sess, children, **kwargs)
    
    def calc_result(self, a):
        return np.expand_dims(a, axis=self.axis)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.squeeze(grad, op.axis)]
    
    def calc_shape(self, a):
        res = list(a)
        a = self.axis
        return tuple(res[:a] + [1] + res[a:])
    
    def calc_name(self, a):
        return 'ExpDims({},{})'.format(a, self.axis)

class SqueezeNode(Node):

    def __init__(self, sess, children, axis, **kwargs):
        self.axis = axis
        super().__init__(sess, children, **kwargs)
    
    def calc_result(self, a):
        return np.squeeze(a, axis=self.axis)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.expand_dims(grad, op.axis)]
    
    def calc_shape(self, a):
        res = list(a)
        del res[self.axis]
        return tuple(res)
    
    def calc_name(self, a):
        return 'Squeeze({})'.format(a)

class ReshapeNode(Node):

    def __init__(self, sess, children, target_shape, **kwargs):
        self.target_shape = target_shape
        super().__init__(sess, children, **kwargs)
    
    def calc_result(self, a):
        self.orig_shape = a.shape
        return np.reshape(a, self.target_shape)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [fl.reshape(grad, op.orig_shape)]
    
    def calc_shape(self, a):
        return self.target_shape
    
    def calc_name(self, a):
        return 'Reshape({},{})'.format(a, self.target_shape)

class AvgNode(Node):

    def __init__(self, sess, children, axis, **kwargs):
        self.axis = axis
        self.num = children[0].shape[axis]
        super().__init__(sess, children, **kwargs)

    def calc_result(self, a):
        return np.average(a, axis=self.axis)

    @staticmethod
    def calc_gradients(op, grad):
        return [fl.repeat(fl.expand_dims(grad, op.axis), op.axis, op.num) / op.num]
    
    def calc_shape(self, a):
        res = list(a)
        x = self.axis
        res = res[:x] + res[x+1:]
        return tuple(res)
    
    def calc_name(self, a):
        return 'Avg({})'.format(a)

class ConcatenateNode(Node):

    def __init__(self, sess, children, axis=0, **kwargs):
        self.axis = axis
        self.alength = None

        ashape = children[0].shape
        sa = [slice(None,None,None) for _ in ashape]
        sb = list(sa)
        sa[axis] = slice(None, ashape[axis], None)
        sb[axis] = slice(ashape[axis], None, None)
        self.selector_a = sa
        self.selector_b = sb

        super().__init__(sess, children, **kwargs)
     
    def calc_result(self, a, b):
        x = self.axis
        self.alength = a.shape[x]
        return np.concatenate((a, b), axis=x)
    
    @staticmethod
    def calc_gradients(op, grad):
        return [grad[op.selector_a], grad[op.selector_b]]
    
    def calc_shape(self, a, b):
        if len(a) != len(b):
            raise Exception('Children of concat should have same length of shape')
        for i in range(len(a)):
            if i != self.axis and a[i] != b[i]:
                raise Exception('Children of concat should have same shape except of concat-axis')
        res = list(a)
        res[self.axis] += b[self.axis]
        return tuple(res)
    
    def calc_name(self, a, b):
        return 'Concat({},{})'.format(a, b)

class SelectNode(Node):

    def __init__(self, sess, children, slices, **kwargs):
        self.slices = slices
        self.ashape = children[0].shape
        super().__init__(sess, children, **kwargs)
    
    def calc_result(self, a):
        return a[self.slices]
    
    @staticmethod
    def calc_gradients(op, grad):
        g = np.zeros(op.ashape)
        g[op.slices] = grad
        return [g]
    
    def calc_shape(self, a):
        # TODO: advance it
        return np.empty(a)[self.slices].shape
    
    def calc_name(self, a):
        return 'Select({})'.format(a)
        
class TransposeNode(Node):

    def calc_result(self, a):
        return a.T
    
    @staticmethod
    def calc_gradients(op, grad):
        return [grad.T]

    def calc_shape(self, a):
        return a[::-1]
    
    def calc_name(self, a):
        return 'Transpose({})'.format(a)
    
class Conv2DNode(Node):

    def calc_result(self, a, b):
        self.filter_wh = b.shape[2:]
        return mult_conv2d(a, b)
    
    @staticmethod
    def calc_gradients(op, grad):
        g = fl.conv2d_gradient(grad, op.children[0], self.filter_wh)
        return [None, g]
    
    def calc_shape(self, a, b):
        if a[1] != b[1]:
            raise Exception('Conv2D: not proper filter in_channel size (2nd dim)')
        return (a[0], b[0], a[2] - b[2] + 1, a[3] - b[3] + 1)
    
    def calc_name(self, a, b):
        return 'Conv2D({},{})'.format(a, b)

class Conv2DGradientNode(Node):

    def __init__(self, sess, children, filter_wh, **kwargs):
        self.filter_wh = filter_wh
        super().__init__(sess, children, **kwargs)

    def calc_result(self, grad, a):
        return mult_conv2d_gradient(grad, a, self.filter_wh)
    
    def calc_shape(self, a, b):
        return (grad[1], a[1], self.filter_wh[0], self.filter_wh[1])
    
    def calc_name(self, a, b):
        return 'Conv2DGrad({},{})'.format(a, b)
