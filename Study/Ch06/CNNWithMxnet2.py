from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)


# ctx = mx.gpu()
ctx = mx.cpu()

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, axes=(2, 0, 1))
    data = data.astype(np.float32)
    return data, label


batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train = True, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')


for d, l in train_data:
    break

print(d.shape, l.shape)

d.dtype




model.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))

input_shape = (224, 224, 3)
alex_net = gluon.nn.Sequential()
with alex_net.name_scope():
    # 1층
    alex_net.add(gluon.nn.Conv2D(channels = 96, kernel_size = 11, strides = 4, padding = 'same',
                                 input_shape = input, activation = 'relu'))
    # 2층
    alex_net.add(gluon.nn.Conv2D(channels = 256, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size = 3, strides = 2))
    # 정규화
    # alex_net.add(mx.nd.L2Normalization(input_shape = alex_net, out = [1:,]))
    # 3층
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, padding='same', activation='relu'))
    # 정규화
    # alex_net.add(mx.nd.L2Normalization(input_shape = alex_net, out = [1:,]))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    # 4층
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, padding='same', activation='relu'))
    # 5층
    alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, padding='same', activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size = 3, strides = 2))

    # 완전연결층
    # 평면화
    alex_net.add(gluon.nn.Flatten())
    #6층
    alex_net.add(gluon.nn.Dense(4096, activation = 'relu'))
    alex_net.add(mx.gluon.nn.Dropout(rate=0.5))
    # 7층
    alex_net.add(gluon.nn.Dense(4096, activation = 'relu'))
    alex_net.add(mx.gluon.nn.Dropout(rate=0.5))
    # 8층=출력층
    alex_net.add(gluon.nn.Dense(1000, activation = 'softmax'))

# 초기값 설정
alex_net.collect_params().initialize(mx.init.Xavier(magnitude = 2.24), ctx = ctx)


trainer = gluon.Trainer(alex_net.collect_params(), 'sgd', {'learning_rate': 0.01})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis = 1)
        acc.update(preds = predictions, labels = label)
    return acc.get()

#######
# Only one epoch so tests can run quickly, increase this variable to actually run
#######

epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = alex_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ############
        # keep a moving average of the losses
        ############

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, alex_net)
    train_accuracy = evaluate_accuracy(train_data, alex_net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))


