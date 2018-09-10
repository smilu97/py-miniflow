import numpy as np
from flow.node import Node

class VariableNode(Node):

    def __init__(self, sess, value):
        super().__init__(sess, [])
        self.result = value

    def apply_gradient(self):
        self.result -= self.gradient * self.sess.lr

class PlaceholderNode(Node):

    def __init__(self, sess, value, name):
        super().__init__(sess, [])
        self.result = value
        self.name = name
        sess.register_placeholder(self)
        
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

class AddNode(Node):

    def calc_result(self, a, b):
        return a + b

    def calc_gradients(self):
        return [self.gradient, self.gradient]

class SubNode(Node):

    def calc_result(self, a, b):
        return a - b

    def calc_gradients(self):
        return [self.gradient, -(self.gradient)]

class SigmoidNode(Node):

    def calc_result(self, a):
        return 1.0 / (1 + np.exp(-a))

    def calc_gradients(self):
        return [self.result * (1 - self.result) * self.gradient]

class SquareNode(Node):

    def calc_result(self, a):
        return a * a

    def calc_gradients(self):
        v0 = self.children[0].get_result(self.result_version)
        return [2 * v0 * self.gradient]

class TileNode(Node):

    def __init__(self, sess, children, num, axis):
        super().__init__(sess, children)
        self.num = num
        self.axis = axis

    def calc_result(self, a):
        return np.repeat(np.expand_dims(a, self.axis), self.num, self.axis)

    def calc_gradients(self):
        return [np.sum(self.gradient, axis=self.axis)]

class ReduceSumNode(Node):

    def __init__(self, sess, children, axis):
        super().__init__(sess, children)
        self.axis = axis

    def calc_result(self, a):
        self.num = a.shape[self.axis]
        return np.sum(a, axis=self.axis)

    def calc_gradients(self):
        return [np.repeat(np.expand_dims(self.gradient, self.axis), self.num, axis=self.axis)]
