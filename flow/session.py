import numpy as np

class Session:

    def __init__(self, lr=0.001):
        self.nodes = []
        self.placeholders = {}
        self.trainable_nodes = []
        self.lr = lr
    
    def register_node(self, node):
        self.nodes.append(node)
        if hasattr(node, 'apply_gradient'):
            self.trainable_nodes.append(node)
    
    def register_placeholder(self, node):
        self.placeholders[node.name] = node
    
    def set_placeholder(self, name, value):
        self.placeholders[name].result = value
        
    def clean_gradients(self):
        for node in self.nodes:
            node.gradient = None
            node.numGradient = 0
    
    def apply_gradients(self):
        for node in self.trainable_nodes:
            node.apply_gradient()
            