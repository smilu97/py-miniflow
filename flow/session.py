import numpy as np

class Session:

    def __init__(self):
        self.nodes = []
        self.placeholders = {}
        self.trainable_nodes = []
    
    def register_node(self, node):
        self.nodes.append(node)
        if node.trainable:
            self.trainable_nodes.append(node)
    
    def register_placeholder(self, node):
        self.placeholders[node.name] = node
    
    def set_placeholder(self, name, value):
        self.placeholders[name].result = value
        
    def clean_gradients(self):
        for node in self.nodes:
            node.gradient = 0
            node.numGradient = 0
            