import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import math
from sklearn.utils import shuffle


class MultiLayerPerceptron:
    # 2층 퍼셉트론 예시
    def __init__(self, learning_rate=0.01, batch_size=32, hidden_node=10):
        self.w1 = None  # 은닉층 가중치 행렬
        self.b1 = None  # 은닉층 바이어스 배열
        self.w2 = None  # 출력층 가중치 행렬
        self.b2 = None  # 출력층 바이어스 배열
        self.lr = learning_rate  # 학습률
        self.batch_size = batch_size  # 한꺼번에 처리할 튜플의 개수
        self.node = hidden_node  # 은닉층 노드 개수
        self.losses = []  # 손실(훈련세트)
        self.val_losses = []  # 손실(검증세트)
        self.accuracy = []  # 정확도(훈련세트)
        self.val_accuracy = []  # 정확도(검증세트)

    def init_weights(self, m, n):
        # 가중치 초기화
        k = self.node
        # 가우시안 분포로 입력층과 은닉층 사이의 가중치 초기화
        # 가로 m 세로 k 크기의 행렬을 0~1 사이의 랜덤값으로 초기화
        # 입력값 개수 m, 은닉층 노드 개수 k, 출력값 개수 n
        # 바이어스는 전부 0으로 초기화
        self.w1 = np.random.normal(0, 1, (m, k))
        self.b1 = np.zeros(k)
        self.w2 = np.random.normal(0, 1, (k, n))
        self.b2 = np.zeros(n)

    def sigmoid(self, z1):
        # 시그모이드 액티베이션함수 -  입력층과 은닉층 사이
        # a1 = 은닉층 노드 값의 모음(행렬)
        z1 = np.clip(z1, -100, None)
        a1 = 1 / (1 + np.exp(-z1))
        return a1

    def softmax(self, z2):
        # softmax 액티베이션 함수 - 은닉층과 출력층 사이
        # a2 = 출력층 노드 값의 모음(행렬)
        z2 = np.clip(z2, -100, None)
        exp_z2 = np.exp(z2)
        a2 = exp_z2 / np.sum(exp_z2, axis=1).reshape(-1, 1)
        return a2

    def forward1(self, x):
        # 입력층 -> 은닉층 순전파
        # 입력층과 첫번째 가중치(입력-은닉)의 행렬곱을 구함
        z1 = np.dot(x, self.w1) + self.b1
        return z1

    def forward2(self, a1):
        # 은닉층 -> 출력층 순전파
        # 은닉층과 두번째 가중치(은닉-출력)의 행렬곱을 구함
        z2 = np.dot(a1, self.w2) + self.b2
        return z2

    def loss(self, x, y):
        # 손실 계산 (z1과 z2는 액티베이션 함수를 통과하기 전의 가중치행렬*노드행렬 + 바이어스항)
        # 가중치행렬*노드행렬 = 각각의 가중치*노드값의 모음
        z1 = self.forward1(x)
        a1 = self.sigmoid(z1)
        z2 = self.forward2(a1)
        a2 = self.softmax(z2)
        a2 = np.clip(a2, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(a2))  # 크로스 엔트로피

    def backpropagation(self, x, y):
        # 역전파 알고리즘
        z1 = self.forward1(x)
        a1 = self.sigmoid(z1)
        z2 = self.forward2(a1)
        a2 = self.softmax(z2)

        w2_grad = np.dot(a1.T, -(y - a2))  # 은닉층 가중치의 기울기 계산
        b2_grad = np.sum(-(y - a2))  # 은닉층 바이어스의 기울기 계산

        # 은닉층으로 전파된 오차 계산
        hidden_err = np.dot(-(y - a2), self.w2.T) * a1 * (1 - a1)

        w1_grad = np.dot(x.T, hidden_err)  # 입력층 가중치의 기울기 계산
        b1_grad = np.sum(hidden_err, axis=0)  # 입력층 바이어스의 기울기 계산

        return w1_grad, b1_grad, w2_grad, b2_grad

    def minibatch(self, x, y):
        iteration = math.ceil(len(x) / self.batch_size)

        x, y = shuffle(x, y)

        for i in range(iteration):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]

    def fit(self, x_data, y_data, epochs=40, x_val=None, y_val=None):
        # 학습 함수
        self.init_weights(x_data.shape[1], y_data.shape[1])  # 가중치 초기화
        for epoch in range(epochs):
            cumulated_loss = 0  # 손실 누적할 변수
            # 가중치 누적할 변수들 초기화
            for x, y in self.minibatch(x_data, y_data):
                cumulated_loss += self.loss(x, y)  # 손실 누적
                # 역전파 알고리즘을 이용하여 각 층의 가중치와 바이어스 기울기 계산
                w1_grad, b1_grad, w2_grad, b2_grad = self.backpropagation(x, y)

                # 은닉층 가중치와 바이어스 업데이트
                self.w2 -= self.lr * w2_grad / len(x)
                self.b2 -= self.lr * b2_grad / len(x)

                # 입력층 가중치와 바이어스 업데이트
                self.w1 -= self.lr * w1_grad / len(x)
                self.b1 -= self.lr * b1_grad / len(x)

            # 검증 손실 계산
            val_loss = self.val_loss(x_val, y_val)

            self.losses.append(cumulated_loss / len(y_data))
            self.val_losses.append(val_loss)
            self.accuracy.append(self.score(x_data, y_data))
            self.val_accuracy.append(self.score(x_val, y_val))

            print(f'epoch({epoch + 1}) ===> loss : {cumulated_loss / len(y_data):.5f} | val_loss : {val_loss:.5f}',
                  f' | accuracy : {self.score(x_data, y_data):.5f} | val_accuracy : {self.score(x_val, y_val):.5f}')

    def predict(self, x_data):
        z1 = self.forward1(x_data)
        a1 = self.sigmoid(z1)
        z2 = self.forward2(a1)
        return np.argmax(z2, axis=1)  # 가장 큰 인덱스 반환

    def score(self, x_data, y_data):
        return np.mean(self.predict(x_data) == np.argmax(y_data, axis=1))

    def val_loss(self, x_val, y_val):  # 검증 손실 계산
        val_loss = self.loss(x_val, y_val)
        return val_loss / len(y_val)


# mnist 데이터를 받아옴
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 검증 세트 생성
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)

# 28 x 28 배열을 1 x 784 배열로 바꿈
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_val = x_val.reshape(x_val.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# 0~255 값 분포를 0~1 사이에 분포하도록 바꿈 (정규화)
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# one-hot encoding
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)
print(math.ceil(len(x_train) / 256))

# 학습 시작
model = MultiLayerPerceptron(learning_rate=0.1, batch_size=256, hidden_node=100)
model.fit(x_train, y_train, epochs=40, x_val=x_val, y_val=y_val)
model.score(x_test, y_test)
