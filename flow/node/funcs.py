from flow.node import *

def matmul(a, b):
    return MatmulNode(a.sess, [a, b])

def add(a, b):
    return AddNode(a.sess, [a, b])

def sub(a, b):
    return SubNode(a.sess, [a, b])

def sigmoid(a):
    return SigmoidNode(a.sess, [a])

def square(a):
    return SquareNode(a.sess, [a])

def transpose(a):
    return TransposeNode(a.sess, [a])

def concat(a, b, axis=0):
    return ConcatenateNode(a.sess, [a, b], axis)

def sum(a, axis):
    return SumNode(a.sess, [a], axis)