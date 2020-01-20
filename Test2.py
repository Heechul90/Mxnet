## 예제 실행해 보기

import mxnet as mx
from mxnet import nd


mx.random.seed(1)
x = nd.empty((3, 4))
print(x)