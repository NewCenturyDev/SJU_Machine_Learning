import numpy as np


class Perceptron:
    def __init__(self, **kwargs):
        self.input_vector = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.label = np.array([-1, 1, 1, 1])
        # 각각의 가중치를 1로 초기화 (w_0은 배열 생성시 이미 1로 초기화됨)
        self.weight = np.array([1., 1., 1.])

        if 'x' in kwargs:
            self.input_vector = kwargs['x']
        if 'y' in kwargs:
            self.label = kwargs['y']
        if 'weight' in kwargs:
            self.weight = kwargs['weight']

    def forward(self, x):
        # 입력값의 모든 값에 대해 각각의 가중치 값을 모두 곱해 더하고, 바이어스항 역시 더해 줌
        result = self.weight[0]
        for i in range(2):
            result += x[i] * self.weight[i + 1]
        return result
        # 반복문을 간단하게 하기 위해 행렬 X에 각각의 weight(w1~w4)를 행렬로 변환
        # 가중치를 행렬로 취급하여 가중치 행렬을 입력값 행렬에 행렬곱을 하고, 바이어스항을 더하는 방식으로도 가능
        # 위 연산을 줄인 것이 아래의 것이다
        # return np.dot(x, self.weight[1:]) + self.weight[0]

    def predict(self, x):
        # 학습된 데이터로 조건을 만족하는 데이터에 대한 class 예측
        # forward 계산을 하여 0보다 크면 1, 0보다 작으면 -1을 반환함
        # 스텝 펑션을 적용했다고도 볼 수 있음
        return np.where(self.forward(x) > 0, 1, -1)

    def do_train(self):
        for iteration in range(50):
            for x_val, y_val in zip(self.input_vector, self.label):
                update = 0.01 * (y_val - self.predict(x_val))
                self.weight[1:] += update * x_val
                self.weight[0] += update
        print('Predicted: ')
        print(self.weight)


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
OR_Y = np.array([-1, 1, 1, 1])
AND_Y = np.array([-1, -1, -1, 1])
# 각각의 가중치를 1로 초기화 (w_0은 배열 생성시 이미 1로 초기화됨)
W = np.array([1., 1., 1.])

# OR 분류기 퍼셉트론
Perceptron(x=X, y=OR_Y, weight=W).do_train()
# AND 분류기 퍼셉트론
Perceptron(x=X, y=AND_Y, weight=W).do_train()
