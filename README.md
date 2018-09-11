
# Miniflow

## What is miniflow?

Miniflow is a implementing project of tensorflow-like functions, utilities just for studying.
It may be a very-easy, most basic version of Deep-learning framework

## How to try

```sh
git clone https://github.com/smilu97/miniflow
cd miniflow
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
python test.py
```

## Learning XOR Test

![xor_test](static/xor_test.png)

Simple logistic regression

- 2 Input (x)
- 2 HiddenLayer (S0 = sigmoid(x * V0 + b0))
- 1 Output (S1 = sigmoid(S0 * V1 + b1))
  
### [Tensorflow-like graph building](test/xor.py)

```python
sess = fl.Session(lr=0.1)

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
```

## TODO

- [x] Basic Graph Node
- [x] XOR learning test
- [x] Concat, Select - not tested
- [x] Transpose
- [x] Shape validations
- [ ] Convolution2D, MaxPool, AvgPool etc...
