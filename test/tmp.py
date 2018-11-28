import numpy as np
import flow as fl

def test():
    sess = fl.Session()
    a = fl.ones(sess, (10,))
    opt = fl.SampleOptimizer(sess)
    mini = opt.minimize(a)
    for _ in range(10):
        sess.run([mini])
        print(a.result)