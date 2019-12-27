import mxnet as mx
from mxnet import gluon, autograd, ndarray


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


def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model