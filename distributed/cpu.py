import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn



##################
# Hyperparameter #
#----------------#
ctx = mx.cpu()
lr=0.05
epochs=10
momentum=0.9
batch_size=200
kv=mx.kv.create("dist_sync")
#----------------#
# Hyperparameter #
##################



# data
mnist = mx.test_utils.get_mnist()
train_data=gluon.data.ArrayDataset(mnist['train_data'], mnist['train_label'])
val_data=gluon.data.ArrayDataset(mnist['test_data'], mnist['test_label'])
train_data = gluon.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = gluon.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)



# define network
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels=32, kernel_size=(3, 3), activation='relu'))
    net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(nn.Conv2D(channels=64, kernel_size=(3, 3), activation='relu'))
    net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10, activation=None))



# train
def train():
    net.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer='sgd',
                            optimizer_params={'learning_rate': lr, 'momentum': momentum},
                            kvstore=kv,
                            update_on_kvstore=True)

    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    
    stride=int(batch_size/kv.num_workers)
    start=kv.rank*stride
    end=start+stride
    if kv.rank==kv.num_workers:
       end=batch_size

    for epoch in range(epochs):
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            if len(data)<start:
                trainer.step(batch_size)
                break
            elif len(data)<batch_size:
                end=len(data)

            data = data[start:end].as_in_context(ctx)
            label = label[start:end].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            trainer.step(data.shape[0])
            metric.update([label], [output])

        name, acc = metric.get()
        if kv.rank==0:
            print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))



# test
def test():
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    name, acc = metric.get()
    print('Validation: %s=%f'%(name, acc))



if __name__ == '__main__':
    train()
    if kv.rank==0:
        test()