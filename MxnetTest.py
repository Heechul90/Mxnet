import mxnet as mx
import numpy as np

## 1차원 배열
a = mx.nd.array([1, 2, 3])
print(a)
print(a.shape)
print(a.dtype)
print(a.asnumpy())

## 2차원 배열
b = mx.nd.array([[1, 2, 3], [2, 3, 4]])
print(b)
print(b.shape)
print(b.dtype)
print(b.asnumpy())


## 타입 설정해서 배열 생성하기
a = mx.nd.array([1, 2, 3], dtype = np.int32)
print(a.dtype)


## 특수 배열 생성자
# 0으로 만든 배열 생성
a = mx.nd.zeros((2, 3))
print(a.shape)
print(a.asnumpy())

# 1로 만든 배열 생성
a = mx.nd.ones((2, 3))
print(a.shape)
print(a.asnumpy())

# 7로 만든 배열 생성
a = mx.nd.full((2, 3), 7)
print(a.shape)
print(a.asnumpy())

## 배열 요소 단위의 기본 오퍼레이션
a = mx.nd.ones((2, 3))
b = mx.nd.full((2, 3), 5)
c = a + b
print(c.shape)
print(c.asnumpy())

d = a - b
print(d.shape)
print(d.asnumpy())


## 접근 및 슬라이스
# indexing
a = mx.nd.array(np.arange(6).reshape(3, 2))
print(a.shape)
print(a.asnumpy())
print(a[0][1].asnumpy())

# slicing
print(a[1:2].asnumpy())

# 옵션주기 행으로 1, 2
d = mx.nd.slice_axis(a, axis = 0, begin = 1, end = 2)

# axis = 1로
e = mx.nd.slice_axis(a, axis = 1, begin = 1, end = 2)

## shape 변경하기
a = mx.nd.array(np.arange(24))
print(a.shape)
print(a.asnumpy())

b = a.reshape((2, 3, 4))
print(b.shape)
print(b.asnumpy())

# 값 바꾸기 0번째 채널에 0행, 2컬럼
b[0][0][2] = -9
b[0][0][2]

print(b.asnumpy())
print(a.asnumpy())


## 배열 연결하기
# concatenate
a = mx.nd.ones((2, 3))
b = mx.nd.ones((2, 3)) * 2
print(a.asnumpy())
print(b.asnumpy())

# 밑으로(행으로) 합치기
c = mx.nd.concatenate([a, b])
print(c.asnumpy())

# 옆으로(컬럼으로) 합치기
d = mx.nd.concatenate([a, b], axis = 1)
print(d.asnumpy())


## 리듀스하기
a = mx.nd.ones((2, 3))
print(a.asnumpy())

b = mx.nd.sum(a)
print(b.asnumpy())

# axis에 따라서 합하기
c = mx.nd.sum_axis(a, axis = 0)
print(c.asnumpy())

d = mx.nd.sum_axis(a, axis = 1)
print(d.asnumpy())


## GPU지원
a = mx.nd.ones((100, 100))
b = mx.nd.ones((100, 100), mx.cpu(0))
c = mx.nd.ones((100, 100), mx.gpu(0))    # gpu 안됨
print(a)
print(b)
print(c)


## context 이용
def f():
    a = mx.nd.ones((100, 100))
    b = mx.nd.ones((100, 100))
    c = a + b
    print(c)

# cpu 사용
f()

# gpu 사용
with mx.Context(mx.gpu()):     # gpu 안됨
    f()


