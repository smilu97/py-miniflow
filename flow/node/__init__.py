import numpy as np

class Node:

    def __init__(self, sess, children):
        self.sess = sess
        self.children = children
        self.parentNum = 0
        if not hasattr(self, 'result'):
            self.result = None
        self.result_version = 0
        self.gradient = None
        self.numGradient = 0

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
        if self.gradient == None:
            self.gradient = gradient
        else:
            self.gradient += gradient
    
    def minimize(self):
        result = self.get_result(self.result_version + 1)
        self.sess.clean_gradients()
        self.gradient = np.ones_like(result)
        self.propagate_gradient()
        self.sess.apply_gradients()
    
    def calc_result(self):
        return self.result
    
    def calc_gradients(self):
        return []
    
    def calc_shape(self):
        return None