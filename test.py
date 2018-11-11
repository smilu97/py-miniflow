#!/usr/bin/env python3

import sys
from test.sin import test as sin_test
from test.xor import test as xor_test
from test.logistic import test as logistic_test
from test.mnist_cnn import test as mnist_test

tests = [
    ['sin', sin_test],
    ['xor', xor_test],
    ['logistic', logistic_test],
    ['mnist', mnist_test],
]

if len(sys.argv) > 1:
    name = sys.argv[1]
    found = False
    for test in tests:
        if test[0] == name:
            test[1]()
            found = True
            break
    if found:
        exit(0)

for idx, test in enumerate(tests):
    print('{}) {} test'.format(idx + 1, test[0]))
try:
    sel = int(input()) - 1
    if type(sel) is type(1) and 0 <= sel < len(tests):
        tests[sel][1]()
except:
    print('Invalid input')
