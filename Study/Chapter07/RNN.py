from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
import pandas as pd
mx.random.seed(1)
ctx = mx.cpu()

dataset = pd.read_csv('dataset/crack1.csv',
                      index_col = 0)
data = dataset.copy()
data.head(2)

with open('timemachine.txt') as f:
    time_machine = f.read()

print(time_machine[0:500])

print(time_machine[-38075:37500])
time_machine = time_machine[:38083]

character_list = list(set(time_machine))
vocab_size = len(character_list)
print(character_list)
print('Length of vocab: %s' % vocab_size)

character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
print(character_dict)

##################################
# Check that the length is right
##################################
print(len(time_numerical))

##################################
# Check that the format looks right
##################################
print(time_numerical[:20])

##################################
# Convert back to text
##################################
print("".join([character_list[idx] for idx in time_numerical[:39]]))

def one_hots(numerical_list, vocab_size = vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx = ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

print(one_hots(time_numerical[:2]))

def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

textify(one_hots(time_numerical[0:40]))

seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
textify(dataset[0])

batch_size = 32

