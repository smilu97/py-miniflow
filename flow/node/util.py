import numpy as np

def xavier(*shape, uniform=False):
    fan_in = shape[0]
    fan_out = shape[-1]
    x = np.sqrt((6 if uniform else 2) / (fan_in + fan_out))
    return (np.random.rand if uniform else np.random.randn)(*shape) * x

def shape_broadcast(s0, s1):
    res = []
    if len(s0) < len(s1):
        s0, s1 = s1, s0
    l0 = len(s0)
    l1 = len(s1)
    dl = l0 - l1
    for i in range(dl):
        res.append(s0[i])
    for i in range(l1):
        e0 = s0[dl + i]
        e1 = s1[i]
        if e0 == e1 or e0 == 1 or e1 == 1:
            res.append(max(e0, e1))
        else:
            raise Exception('Shape broadcasting error: {}, {}'.format(s0, s1))
    return tuple(res)

def array_fit_to_shape(a, shape):
    if len(a.shape) < len(shape):
        raise Exception('Fitting array to shape error: {}, {}'.format(a.shape, shape))
    asl = len(a.shape) # a.shape.length -> asl
    sl = len(shape)
    dl = asl - sl
    for i in range(dl):
        a = np.sum(a, 0)
    for i in range(sl):
        if a.shape[i] != shape[i] and shape[i] != 1:
            raise Exception('Fitting array to shape error: {}, {}'.format(a.shape, shape))
        if shape[i] == 1:
            a = np.sum(a, i)
            a = np.expand_dims(a, i)
    return a