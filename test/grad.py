import flow as fl
import numpy as np

def test():
    sess = fl.Session()
    a = fl.ones(sess, (1,))
    b = fl.ones(sess, (1,))
    c = a + b
    grads = fl.gradients([c], [a, b])
    print(grads[0].get_result())

    b = a * 2
    grads = fl.gradients([b], [a])
    print(grads[0].get_result())

    b = a + a
    grads = fl.gradients([b], [a])
    print(grads[0].get_result())

    # 2a^2 + 3a + 4
    # 4a + 3
    b = 2 * fl.square(a) + 3 * a + 4
    grads = fl.gradients([b], [a])
    print(grads[0].get_result())
    print('b:', b.get_name())
    print('g:', grads[0].get_name())

    x = fl.ones(sess, (10, 10), 'x')
    w = fl.ones(sess, (10, 20), 'w')
    mul = fl.matmul(x, w, 'mul')
    y_ = fl.sigmoid(mul, 'sigmoid')
    y = fl.ones(sess, (10, 20), 'y_')
    e = fl.l2loss(y_, y)
    grads = fl.gradients([e], [w])
    print('g:', grads[0].get_name())
    print('g:', np.sum(grads[0].get_result()))