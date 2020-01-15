# Mxnet

# Mxnet을 활용한 신경망 

<img src="http://image.yes24.com/Goods/69730346/800x0">


---

이 저장소는 『[밑바닥부터 시작하는 딥러닝](http://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)』(한빛미디어, 2017)의 지원 사이트입니다. ([『밑바닥부터 시작하는 딥러닝 ❷』의 깃허브 저장소는 이곳](https://github.com/WegraLee/deep-learning-from-scratch-2)입니다.)


:red_circle: **[공지]** 종종 실습용 손글씨 데이터셋 다운로드 사이트( http://yann.lecun.com/exdb/mnist/ )가 연결되지 않습니다.
그래서 예제 수행에 필요한 데이터셋 파일을 /dataset/ 디렉터리에 올려뒀습니다.
혹 사이트가 다운되어 데이터를 받을 수 없다면 아래 파일 4개를 각자의 <예제 소스 홈>/dataset/ 디렉터리 밑에 복사해두면 됩니다. ^__^

* [t10k-images-idx3-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/t10k-images-idx3-ubyte.gz)
* [t10k-labels-idx1-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/t10k-labels-idx1-ubyte.gz)
* [train-images-idx3-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/train-images-idx3-ubyte.gz)
* [train-labels-idx1-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/train-labels-idx1-ubyte.gz)

---

## 책소개
:최근 4차 산업혁명의 패러다임 변화와 더불어 대내외적으로 제4차 산업혁명 관련 전략과제를 발굴하고 있다. 특히 많은 수학자들을 제4차 산업혁명의 근본적인 연구의 주요 연구자로 흡입하는 정책을 펼치는 등 수학의 역할이 점차적으로 산업 및 경제에 기여하는 바가 커지고 있다. 인공신경망을 공부하는 학생들이 텐서플로우, 테아노, 토치, 카페 등과 같은 다양한 딥러닝 프레임워크와 비교하여 장점이 훨씬 더 많은 MXNet을 선택하여 공부하는 것이 더 효율적일 것이다.

이러한 상황에서 공학이나 자연과학을 공부하는 학생들에게 MXNet을 활용한 인공신경망 학습 교재를 개발하여 산업체, 학계, 연구소 등에서 필요한 신경망 및 Deep Learning을 위한 수학적 백그라운드와 프로그램 개발 등을 위한 전문 도서를 집필하고자 한다. 딥러닝 기술의 진보와 다양한 영역에 기계 학습용 프로그래밍 도구들이 개발되고 있으나, 산업계 및 학계에 널리 알려지지 않은 최신 기계 학습 도구이자 아마존이 딥러닝 플랫폼으로 선택한 MXNet의 사용법과 프로그램 개발방법을 소개하고자 한다. 최근에 개발된 학습 도구인 MXNet이 앞으로 가장 널리 사용될 프레임에도 불구하고 이를 활용한 교재는 아직 전무한 상태이므로 본 교재는 매우 유용하게 사용될 것이다.

## 책 미리보기
[issuu](https://issuu.com/hanbit.co.kr/docs/____________________________________38d0e6451f0ddf) | [SlideShare](http://www.slideshare.net/wegra/ss-70456623) | [Yumpu](https://www.yumpu.com/xx/document/view/56594155/-)

## 목차

Chapter 1. 파이썬 배우기 / 11
Chapter 2. 파이썬으로 풀어보는 수학 / 22
Chapter 3. 신경망의 이해 / 42
Chapter 4. 신경망의 이해 Ⅱ / 58
Chapter 5. MXNet 소개 및 설치하기 / 74
Chapter 6. MXNet를 이용한 CNN / 120
Chapter 7. MXNet를 이용한 RNN / 129
Chapter 8. MXNet를 이용한 컴퓨터 비전 / 150
Chapter 9. TDA (Topology Data Analysis) / 174

참고사항 - H/W Spec의 문제 / 197
참고문헌 / 199


## 책 속으로
Chapter 1 파이썬 배우기

파이썬 설치하기

파이썬을 설치하기 위해서는 먼저, 파이썬의 공식 사이트(https://www.python.org/)에 접속한 후에, 사용 중인 시스템의 운영체제에 맞는 파이썬 버전을 다운로드 및 설치를 진행한다. 여기서는 윈도우 운영체제의 python 3.6.5 버전을 선택한다. 참고로, 파이썬 패키지 453개 정도를 포함하고 있는 아나콘다 배포판을 통하여 파이썬을 설치할 수도 있다.
(https://repo.continuum.io/archive/)

설치가 완료되면 윈도우의 시작 메뉴에서 Python 메뉴를 볼 수 있으며, 메뉴 중에서 IDLE (Python 3.6 64-bit)를 선택한다.

그러면, 다음의 그림과 같이 Python 3.6.5 Shell이 나타나게 되며, Python의 설치가 완료된 것이다.

파이썬 쉘을 이용하여 다음과 같은 기본적인 연산을 수행하여 본다.

파이썬의 matplotlib

파이썬의 matplotlib 모듈은 그래프를 만드는데 유용하게 사용된다. 윈도우의 DOS 커맨드 창에서 다음의 명령어를 입력하여 matplotlib 모듈을 설치한다.

〉 python -m pip install matplotlib

x 변수와 y 변수에 값들을 할당한 후에 이 값들을 matplotlib 모듈을 이용하여 그래프로 나타내고자 한다. 파이썬 쉘에서 다음과 같이 소스 코드를 입력하면 그래프로 확인이 가능하게 된다.

Source Code
import matplotlib
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
from pylab import plot, show
plot(x, y)
show()

앞부분에서 x 변수와 y 변수에 값들을 할당한 후, 이 값들을 matplotlib 모듈을 이용하여 그래프로 나타내었다. 이번에는 그래프를 이미지로 저장하고자 하며, 파이썬 쉘에서 다음과 같이 소스 코드를 입력하면 이미지가 저장된다. 이미지가 저장된 폴더에서 저장된 Figure1.png 이미지의 확인이 가능하다.

Source Code
import matplotlib
from pylab import plot, savefig
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 4, 2]
plot(x, y)
savefig('C:\Figure1.png')


파이썬의 수학 함수

파이썬 표준 라이브러리 math에서 일반 수학 함수를 사용할 수 있으며, 다음의 파이썬 코드에서 삼각함수 sin 값과 cos 값을 출력하게 된다.

Source Code
import math
math.sin(math.pi/2)
math.cos(math.pi/2)


파이썬의 Numpy 라이브러리와 파이썬의 Scipy 라이브러리

일반수학 함수 외에도 고급수학 함수를 이용하기 위한 파이썬 표준 라이브러리 Numpy와 Scipy를 윈도우즈 커맨드 창에서 아래와 같이 입력하여 설치한다.

〉 python -m pip install numpy
〉 python -m pip install scipy

파이썬 표준 라이브러리 Numpy는 수치해석, 특히 선형대수(linear algebra) 계산을 위한 패키지이다. 자료형이 고정된 다차원 배열 클래스(n-dimensional array)와 벡터화 연산(vectorized operation)을 지원하며 수학 연산을 위한 패키지이다. 또한, 파이썬 표준 라이브러리 Scipy는 고급수학 함수, 수치적 미적분, 미분방정식 계산, 최적화, 신호 처리 등을 위한 다양한 과학기술 계산 기능을 제공하는 패키지이다.
설치 완료된 파이썬 표준 라이브러리 Numpy와 Scipy를 이용하여 간단하게 동작 여부를 확인한다. 먼저, 파이썬 표준 라이브러리 Numpy의 동작 여부를 확인하기 위하여 다음의 파이썬 코드를 커맨드 파이썬 쉘과 IDE 파이썬 쉘에 입력하여 본다.

Source Code
from numpy import *
random.rand(10)
random.rand(10,1)

파이썬 표준 라이브러리 Numpy의 동작 여부를 확인하였으며, 이번에는 파이썬 표준 라이브러리 Scipy의 동작 여부를 확인하기 위하여 다음의 파이썬 코드를 입력한다.

Source Code
from scipy.fftpack import fft
import numpy as np

N = 600

T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

Result

파이썬 표준 라이브러리 Scipy의 하위 패키지 special을 이용하여 다음의 파이썬 코드로 3D 이미지를 생성하여 본다.

Source Code

from scipy import special
import numpy as np
def drumhead_height(n, k, distance, angle, t):
kth_zero = special.jn_zeros(n, k)[-1]
return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)
theta = np.r_[0:2*np.pi:50j]
radius = np.r_[0:1:50j]
x = np.array([r * np.cos(theta) for r in radius])
y = np.array([r * np.sin(theta) for r in radius])
z = np.array([drumhead_height(1, 1, r, theta, 0.5) for r in radius])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
