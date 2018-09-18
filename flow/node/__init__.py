import numpy as np
from flow.node.util import *

class Node:

    def __init__(self, sess, children, trainable=False):
        self.sess = sess
        self.children = children
        self.parentNum = 0
        if not hasattr(self, 'result'):
            self.result = None
        self.result_version = 0
        self.gradient = None
        self.numGradient = 0
        self.trainable = trainable
        self.initializer_props = None

        for child in children:
            child.parentNum += 1
        
        sess.register_node(self)

        self.shape = self.calc_shape(*[child.shape for child in children])
    
    def get_name(self):
        if (not hasattr(self, 'name')) or self.name is None:
            self.name = self.calc_name(*[child.get_name() for child in self.children])
        return self.name

    def get_result(self, version=None):
        if version == None:
            version = self.result_version + 1
        if self.result_version != version:
            children_results = [child.get_result(version) for child in self.children]
            self.result = self.calc_result(*children_results)
            self.result_version = version
        return self.result
    
    def get_children_result(self):
        return (child.get_result(self.result_version) for child in self.children)
    
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
    
    def __truediv__(self, a):
        a = self.check_transform_constant(a)
        return DivNode(self.sess, [self, a])
    
    def __neg__(self):
        return NegNode(self.sess, [self])

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

    def calc_gradients(self):
        # ab bc ac
        v0 = self.children[0].get_result(self.result_version)
        v1 = self.children[1].get_result(self.result_version)
        g0 = np.matmul(self.gradient, v1.T)
        g1 = np.matmul(v0.T, self.gradient)
        return [g0, g1]
    
    def calc_shape(self, a, b):
        if len(a) != 2 or len(b) != 2:
            raise Exception('Child of matmul should be 2-dimensional array')
        return (a[0], b[1])
    
    def calc_name(self, a, b):
        return 'Matmul({},{})'.format(a, b)

class NegNode(Node):
    
    def calc_result(self, a):
        return -a
    
    def calc_gradients(self):
        return [-self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Neg({})'.format(a)

class AddNode(Node):

    def calc_result(self, a, b):
        return a + b

    def calc_gradients(self):
        return [
            array_fit_to_shape(self.gradient, self.children[0].shape),
            array_fit_to_shape(self.gradient, self.children[1].shape)
        ]
    
    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
    
    def calc_name(self, a, b):
        return 'Add({},{})'.format(a, b)

class SubNode(Node):

    def calc_result(self, a, b):
        return a - b

    def calc_gradients(self):
        return [
             array_fit_to_shape(self.gradient, self.children[0].shape),
            -array_fit_to_shape(self.gradient, self.children[1].shape)
        ]
    
    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
    
    def calc_name(self, a, b):
        return 'Sub({},{})'.format(a, b)

class MulNode(Node):

    def calc_result(self, a, b):
        return a * b
    
    def calc_gradients(self):
        return [
            array_fit_to_shape(self.gradient * self.children[0].result, self.children[0].shape),
            array_fit_to_shape(self.gradient * self.children[1].result, self.children[1].shape)
        ]

    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
            
    def calc_name(self, a, b):
        return 'Mul({},{})'.format(a, b)

class DivNode(Node):

    def calc_result(self, a, b):
        return a / b
    
    def calc_gradients(self):
        v0, v1 = self.get_children_result()
        return [
            array_fit_to_shape(self.gradient / v1, v0.shape),
            array_fit_to_shape((-v0) / np.square(v1) * self.gradient, v1.shape)
        ]
    
    def calc_shape(self, a, b):
        return shape_broadcast(a, b)
    
    def calc_name(self, a, b):
        return 'Div({},{})'.format(a, b)

class SigmoidNode(Node):

    def calc_result(self, a):
        return 1.0 / (1 + np.exp(-a))

    def calc_gradients(self):
        return [self.result * (1 - self.result) * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Sigmoid({})'.format(a)

class TanhNode(Node):

    def calc_result(self, a):
        return np.tanh(a)
    
    def calc_gradients(self):
        r = self.result
        return [(1 - r) * (1 + r) * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Tanh({})'.format(a)
    
class ReluNode(Node):

    def calc_result(self, a):
        return np.maximum(a, 0)
    
    def calc_gradients(self):
        v0 = self.children[0].get_result(self.result_version)
        return [np.heaviside(v0, 0) * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Relu({})'.format(a)

class LeakyReluNode(Node):

    def __init__(self, sess, children, alpha):
        self.alpha = alpha
        super().__init__(sess, children)

    def calc_result(self, a, alpha):
        return np.where(a > 0, a, a * self.alpha)
    
    def calc_gradients(self):
        v0 = self.children[0].get_result(self.result_version)
        return [np.where(v0 > 0, 1, self.alpha) * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'LRelu({})({})'.format(self.alpha, a)

class SoftmaxNode(Node):

    def calc_result(self, a):
        self.exps = np.exp(a - np.max(a))
        return exps / np.sum(exps)
    
    def calc_gradients(self):
        exps = self.exps
        expsum = self.sum(exp)
        expsum2 = expsum ** 2
        return [exps * (self.gradient + np.sum(-exps / expsum2 * self.gradient))]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Softmax({})'.format(a)
    
class LogNode(Node):

    def calc_result(self, a):
        return np.log(a)
    
    def calc_gradients(self):
        v0 = self.children[0].get_result(self.result_version)
        return [(1 / v0) * self.gradient]
    
    def calc_shape(self, a):
        return a

    def calc_name(self, a):
        return 'Log({})'.format(a)

class ExpNode(Node):

    def calc_result(self, a):
        return np.exp(a)
    
    def calc_gradients(self):
        return [self.result * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Exp({})'.format(a)
    
class SquareNode(Node):

    def calc_result(self, a):
        return a * a

    def calc_gradients(self):
        v0 = self.children[0].get_result(self.result_version)
        return [2 * v0 * self.gradient]

    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Square({})'.format(a)

class L2LossNode(Node):

    def calc_result(self, a, b):
        self.diff = a - b
        return np.sum(np.square(self.diff) / 2, axis=None)
    
    def calc_gradients(self):
        v0, v1 = self.get_children_result()
        g = self.diff * self.gradient
        return [
            array_fit_to_shape(g, v0.shape),
            -array_fit_to_shape(-g, v1.shape)
        ]

    def calc_shape(self, a, b):
        return (1,)    

    def calc_name(self, a, b):
        return 'L2Loss({},{})'.format(a, b)
    

class SumNode(Node):

    def __init__(self, sess, children, axis, **kwargs):
        self.axis = axis
        super().__init__(sess, children, **kwargs)

    def calc_result(self, a):
        self.num = a.shape[self.axis]
        return np.sum(a, axis=self.axis)

    def calc_gradients(self):
        return [np.repeat(np.expand_dims(self.gradient, self.axis), self.num, axis=self.axis)]
    
    def calc_shape(self, a):
        res = list(a)
        a = self.axis
        res = res[:a] + res[a+1:]
        return tuple(res)
    
    def calc_name(self, a):
        return 'Sum({})'.format(a)

class AvgNode(Node):

    def __init__(self, sess, children, axis, **kwargs):
        self.axis = axis
        super().__init__(sess, children, **kwargs)

    def calc_result(self, a):
        self.num = a.shape[self.axis]
        return np.average(a, axis=self.axis)

    def calc_gradients(self):
        return [np.repeat(np.expand_dims(self.gradient, self.axis), self.num, axis=self.axis) / self.num]
    
    def calc_shape(self, a):
        res = list(a)
        a = self.axis
        res = res[:a] + res[a+1:]
        return tuple(res)
    
    def calc_name(self, a):
        return 'Sum({})'.format(a)

class ConcatenateNode(Node):

    def __init__(self, sess, children, axis=0, **kwargs):
        self.axis = axis
        self.alength = None
        super().__init__(sess, children, **kwargs)
     
    def calc_result(self, a, b):
        x = self.axis
        self.alength = a.shape[x]
        return np.concat(a, b, axis=x)
    
    def calc_gradients(self):
        g = self.gradient
        slices = [slice(None, None) for _ in g.shape]
        slices[axis] = slice(None, self.alength)
        ag = g[tuple(slices)]
        slices[axis] = slice(self.alength, None)
        bg = g[tuple(slices)]
        return [ag, bg]
    
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

    def __init__(self, sess, children, axis, begin, end, **kwargs):
        self.axis = axis
        self.begin = begin
        self.end = end
        self.slices = None
        self.ashape = None
        super().__init__(sess, children, **kwargs)
    
    def calc_result(self, a):
        if self.slices is None:
            self.slices = [slice(None, None) for _ in a.shape]
            self.slices[self.axis] = slice(self.begin, self.end)
            self.ashape = a.shape
        return a[tuple(self.slices)]
    
    def calc_gradients(self):
        g = np.zeros(self.ashape)
        g[tuple(self.slices)] = self.gradient
        return [g]
    
    def calc_shape(self, a):
        res = list(a)
        res[self.axis] = self.end - self.begin
        return tuple(res)
    
    def calc_name(self, a):
        return 'Select({})'.format(a)
        
class TransposeNode(Node):

    def calc_result(self, a):
        return a.T
    
    def calc_gradients(self):
        return [self.gradient.T]

    def calc_shape(self, a):
        return a[::-1]
    
    def calc_name(self, a):
        return 'Transpose({})'.format(a)
    
class Conv2DNode(Node):

    def calc_result(self, a, b):
        self.filter_wh = b.shape[2:]
        return mult_conv2d(a, b)
    
    def calc_gradients(self):
        g = mult_conv2d_gradient(self.gradient, self.children[0].result, self.filter_wh)
        return [None, g]
    
    def calc_shape(self, a, b):
        if a[1] != b[1]:
            raise Exception('Conv2D: not proper filter in_channel size (2nd dim)')
        return (a[0], b[0], a[2] - b[2] + 1, a[3] - b[3] + 1)
    
    def calc_name(self, a, b):
        return 'Conv2D({},{})'.format(a, b)
