import numpy as np
from flow.node import Node

def shape_broadcast(s0, s1):
    res = []
    if len(s0) < len(s1):
        s0, s1 = s1, s0
    l0 = len(s0)
    l1 = len(s1)
    dl = l0 - l1
    for i in range(dl):
        res.append(s0[i])
    for i in range(l1):
        e0 = s0[dl + i]
        e1 = s1[i]
        if e0 == e1 or e0 == 1 or e1 == 1:
            res.append(max(e0, e1))
        else:
            raise Exception('Shape broadcasting error: {}, {}'.format(s0, s1))
    return tuple(res)

def array_fit_to_shape(a, shape):
    if len(a.shape) < len(shape):
        raise Exception('Fitting array to shape error: {}, {}'.format(a.shape, shape))
    asl = len(a.shape) # a.shape.length -> asl
    sl = len(shape)
    dl = asl - sl
    for i in range(dl):
        a = np.sum(a, 0)
    for i in range(sl):
        if a.shape[i] != shape[i] and shape[i] != 1:
            raise Exception('Fitting array to shape error: {}, {}'.format(a.shape, shape))
        if shape[i] == 1:
            a = np.sum(a, i)
            a = np.expand_dims(a, i)
    return a

class VariableNode(Node):

    def __init__(self, sess, value, **kwargs):
        self.result = value
        super().__init__(sess, [], **kwargs)

    def apply_gradient(self):
        self.result -= self.gradient * self.sess.lr
    
    def calc_shape(self):
        return self.result.shape
    
    def calc_name(self):
        return 'Var({})'.format(self.result.shape)

class PlaceholderNode(Node):

    def __init__(self, sess, value, name):
        self.result = value
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
        if a != b:
            raise Exception('Children of sub should have same shape')
        return a
    
    def calc_name(self, a, b):
        return 'Sub({},{})'.format(a, b)

class SigmoidNode(Node):

    def calc_result(self, a):
        return 1.0 / (1 + np.exp(-a))

    def calc_gradients(self):
        return [self.result * (1 - self.result) * self.gradient]
    
    def calc_shape(self, a):
        return a
    
    def calc_name(self, a):
        return 'Sigmoid({})'.format(a)

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
        a[self.axis] = self.end - self.begin
        return tuple(a)
    
    def calc_name(self, a):
        return 'Select({})'.format(a)
        
class TransposeNode(Node):

    def calc_result(self, a):
        return a.T
    
    def calc_gradient(self):
        return [self.gradient.T]

    def calc_shape(self, a):
        return a[::-1]
    
    def calc_name(self, a):
        return 'Transpose({})'.format(a)
