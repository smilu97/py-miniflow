import flow as fl
import numpy as np
import progressbar as pb

def test():

    sess = fl.Session(lr=0.1)

    # Vx + b = y

    train_x = np.array([[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])
    train_y = np.array([[0],[1],[1],[0]])

    x = fl.Placeholder(sess, train_x, 'x')
    
    y = fl.Placeholder(sess, train_y, 'y')

    V0 = fl.Variable(sess, fl.xavier(2,2))
    b0 = fl.Variable(sess, fl.xavier(2))
    S0 = fl.sigmoid(fl.matmul(x, V0) + b0)

    V1 = fl.Variable(sess, fl.xavier(2,1))
    b1 = fl.Variable(sess, fl.xavier(1))
    S1 = fl.sigmoid(fl.matmul(S0, V1) + b1)

    E = fl.sum(fl.square(S1 - y), axis=0)

    print('start error:', E.get_result())

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