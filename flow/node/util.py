import numpy as np

def xavier(shape, fan_in, fan_out, uniform=False):
    x = np.sqrt((6 if uniform else 2) / (fan_in + fan_out))
    res = (np.random.rand if uniform else np.random.randn)(*shape) * x
    return res

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
    d = len(a.shape) - len(shape)
    if d < 0:
        raise Exception('Fitting array to shape error: {}, {}'.format(a.shape, shape))
    if d > 0:
        a = np.sum(a, axis=tuple(range(d)))
    for i in range(len(shape)):
        if a.shape[i] != shape[i]:
            if shape[i] == 1:
                a = np.sum(a, axis=i, keepdims=True)
            # else:
            #     raise Exception('Fitting array to shape error: {}, {}'.format(a.shape, shape))
    return a

def conv2d(a, f):
    '''
    Pure 2d convolution, one channel
    '''
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    subM = np.lib.stride_tricks.as_strided(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def conv3d(a, f):
    '''
    Pure 3d convolution
    '''
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    subM = np.lib.stride_tricks.as_strided(a, shape = s, strides = a.strides * 2)
    return np.einsum('ijk,ijklmn->lmn', f, subM)

def mult_conv2d(data, filters):
    d_shape = data.shape # [batch_size, in_channel, width, height]
    d_strides = data.strides
    f_shape = filters.shape # [out_channel, in_channel, filter_width, filter_height]
    # [batch_size, width - f_width + 1, height - f_height + 1, in_channel, filter_width, filter_height]
    m_shape = (d_shape[0], d_shape[2] - f_shape[2] + 1, d_shape[3] - f_shape[3] + 1, d_shape[1], f_shape[2], f_shape[3])
    # m_shape: (batch_size, shrink_width, shrink_height, in_channel, filter_width, filter_height)
    m_strides = (d_strides[0], d_strides[2], d_strides[3], d_strides[1], d_strides[2], d_strides[3])
    m = np.lib.stride_tricks.as_strided(data, shape = m_shape, strides = m_strides)
    # [batch_size, out_channel, width - filter_width + 1, height - filter_height + 1]
    return np.einsum('jabc,iklabc->ijkl', filters, m)

def mult_conv2d_gradient(gradient, a, filter_wh):
    '''
    gradient: [batch_size, out_channel, width - fwidth + 1, height - fheight + 1]
    '''
    ashape = a.shape
    astrides = a.strides
    # [batch_size, width - f_width + 1, height - f_height + 1, in_channel, filter_width, filter_height]
    m_shape = (ashape[0], ashape[2] - filter_wh[0] + 1, ashape[3] - filter_wh[1] + 1, ashape[1], filter_wh[0], filter_wh[1])
    m_strides = (astrides[0], astrides[2], astrides[3], astrides[1], astrides[2], astrides[3])
    m = np.lib.stride_tricks.as_strided(a, shape=m_shape, strides=m_strides)
    # [out_channel, in_channel, filter_width, filter_height]
    return np.einsum('aibc,abcjkl->ijkl', gradient, m)
