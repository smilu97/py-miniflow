import numpy as np

def save(pathname, weight_vectors):
    fd = open(pathname, 'wb')
    np.savez(fd, **weight_vectors)
    fd.close()

def load(pathname, weight_vectors):
    try:
        fd = open('save.sav', 'rb')
        data = np.load(fd)
        for key in weight_vectors:
            value = weight_vectors[key]
            value.result = data[key]
        fd.close()
    except Exception as e:
        print('Failed to load')