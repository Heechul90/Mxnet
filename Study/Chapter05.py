import mxnet as mx
from mxnet import nd
import numpy as np


def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)

mnist_train = mx.gluon.data.vision.MNIST(train = True, transform = True)

mnist_test = mx.gluon.data.vision.MNIST(train = False, transform = True)

ycount = nd.ones(shape = (10))
xcount = nd.ones(shape = (784, 10))

for data. label in mnist_train:
    x = data.reshape((784, ))
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += x
