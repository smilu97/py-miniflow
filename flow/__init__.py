import numpy as np
from flow.session import Session
from flow.node.base import *
import progressbar as pb

def xor_test():

    sess = Session(lr=0.1)

    # Vx + b = y

    train_x = np.array([[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])
    train_y = np.array([[0],[1],[1],[0]])

    x = PlaceholderNode(sess, train_x, 'x')
    
    y = PlaceholderNode(sess, train_y, 'y')

    V0 = VariableNode(sess, np.random.rand(2,2))
    b0 = VariableNode(sess, np.random.rand(2))
    b0_tiled = TileNode(sess, [b0], 4, 0)
    S0 = SigmoidNode(sess, [AddNode(sess, [MatmulNode(sess, [x, V0]), b0_tiled])])

    V1 = VariableNode(sess, np.random.rand(2,1))
    b1 = VariableNode(sess, np.random.rand(1))
    b1_tiled = TileNode(sess, [b1], 4, 0)
    S1 = SigmoidNode(sess, [AddNode(sess, [MatmulNode(sess, [S0, V1]), b1_tiled])])

    E = ReduceSumNode(sess, [SquareNode(sess, [SubNode(sess, [S1, y])])], 0)

    print('start error:', E.get_result(1))

    epoch = 10000
    with pb.ProgressBar(max_value=epoch) as bar:
        for i in range(epoch):
            E.minimize()
            bar.update(i)
    
    print('last error:', E.get_result())

    print('S1:', S1.get_result())
    print('V0:', V0.result)
    print('b0:', b0.result)
    print('V1:', V1.result)
    print('b1:', b1.result)
