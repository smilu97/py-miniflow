import flow as fl
import numpy as np

class GradientDescentOptimizer:

    def __init__(self, sess, lr=0.001):
        self.sess = sess
        self.lr = lr
    
    def minimize(self, target):
        xs = self.sess.trainable_nodes
        grads = fl.gradients([target], xs)
        for x, grad in zip(xs, grads):
            self.apply_gradient(x, grad.get_result())
        
    def apply_gradient(self, target, grad):
        target.result -= grad * self.lr

class AdamOptimizer(GradientDescentOptimizer):

    def __init__(self, sess, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(sess, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def apply_gradient(self, target, grad):

        # Get previous props
        props = target.initializer_props
        props = {
            'm': 0,
            'v': 0,
            't': 0
        } if props is None else props

        m = props['m']
        v = props['v']
        t = props['t']

        # Prepare gradient and constants
        g = grad
        lr = self.lr
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.epsilon
    
        # Calculate props
        lr_t = lr * np.sqrt((1 - beta2) / (1 - beta1))
        m_t = beta1 * m + (1 - beta1) * g
        v_t = beta2 * v + (1 - beta2) * g * g

        # Apply to result, and update props
        target.result -= lr_t * m_t / (np.sqrt(v_t) + eps)
        target.initializer_props = {
            'm': m_t,
            'v': v_t,
            't': t + 1
        }
