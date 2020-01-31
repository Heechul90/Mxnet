from math import exp
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn
import mxnet as mx
from keras.preprocessing import image
from glob import glob
import cv2, os, random
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils


path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'
dog_path = os.path.join(path, 'dog.1')
len(glob(dog_path))
mx.image.imread(path)