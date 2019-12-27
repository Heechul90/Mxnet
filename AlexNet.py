import mxnet as mx
from mxnet import gluon, autograd, ndarray


# 데이터 가져오기
data = mx.test_utils.get_mnist()

# 학습 데이터 설정 및 사진 재구성
train_data = data['train_data'].reshape((-1, 784))
train_label = data['train_label']



net = gluon.nn.Sequential()
with net.name_scope():
    # 은닉층1
    net.add(gluon.nn.Conv2D(channels=96, kernel_size=(11, 11), padding= 0, strides= 4, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(3, 3), strides=2))
    # 은닉층2
    net.add(gluon.nn.Conv2D(channels=256, kernel_size=(5, 5), padding=1, strides=1, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(3, 3), strides=2))
    # 은닉층3
    net.add(gluon.nn.Conv2D(channels=384, kernel_size=(3, 3), padding=1, strides=1, activation='relu'))
    # 은닉층4
    net.add(gluon.nn.Conv2D(channels=384, kernel_size=(3, 3), padding=1, strides=1, activation='relu'))
    net.Dropout(rate=0.5)
    # 은닉층5
    net.add(gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), padding=1, strides=1, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(3, 3), strides=2))
    # 은닉층6
    net.add(gluon.nn.Conv2D(channels=4096, kernel_size=(6, 6), activation='relu'))
    # 은닉층7
    net.add(gluon.nn.Dense(4096, activation='relu'))
    # 출력층
    net.add(gluon.nn.Dense(1000, activation='softmax'))
