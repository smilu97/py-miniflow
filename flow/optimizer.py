import flow as fl
import numpy as np

class SampleOptimizer:
    '''
    Testing for optimizer symbol
    '''

    def __init__(self, sess, lr=0.001):
        self.sess = sess
        self.lr = fl.Variable(sess, [lr])
    
    def minimize(self, target):
        xs = self.sess.trainable_nodes
        grads = fl.gradients([target], xs)
        return fl.group(*[self._apply_gradient(x, grad) for x, grad in zip(xs, grads)])
    
    def _apply_gradient(self, target, grad):
        return fl.assign(target, target - grad * self.lr)

class GradientDescentOptimizer:

    def __init__(self, sess, lr=0.001):
        self.sess = sess
        self.lr = lr
    
    def minimize(self, target):
        xs = self.sess.trainable_nodes
        grads = target.sess.run(fl.gradients([target], xs))
        for x, grad in zip(xs, grads):
            self.apply_gradient(x, grad)
        
    def apply_gradient(self, target, grad):
        target.result -= grad * self.lr

class AdamOptimizer(GradientDescentOptimizer):

    def __init__(self, sess, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(sess, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr * np.sqrt(1 - beta2) / (1 - beta1)

    def apply_gradient(self, target, grad):

        # Get previous props
        if target.initializer_props is None:
            target.initializer_props = { 'm': 0, 'v': 0 }
        props = target.initializer_props

        m = props['m']
        v = props['v']
    
        # Calculate props
        m_t = self.beta1 * m + (1 - self.beta1) * grad
        v_t = self.beta2 * v + (1 - self.beta2) * grad * grad

        # Apply to result, and update props
        target.result -= self.lr * m_t / (np.sqrt(v_t) + self.epsilon)
        props['m'] = m_t
        props['v'] = v_t
