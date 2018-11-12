import flow as fl

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