import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd

# 데이터 로드
boston = datasets.fetch_openml('boston')
X = boston.data
y = boston.target

# 데이터 스케일링
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 시계열 데이터 생성 함수
# 모델 최적화도 안 되어 있고, 스케일 값도 이상하다
# 결과적으로 학습 결과에 대한 out을 내뱉기 전에 스케일링 한 것을 inverse 시켜야 된다


# LSTM 모델 생성 함수
def create_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    return model


# LSTM 모델 생성
model = create_model()
# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')
# 모델 학습
model.fit(X_train[:, :, np.newaxis], y_train, epochs=50, batch_size=32)
# 모델 평가
mse = model.evaluate(X_test[:, :, np.newaxis], y_test)
print('Mean Squared Error:', mse)

# 예측
predictions = model.predict(X_test[:, :, np.newaxis])


# 정규화 인버스를 반드시 꼭꼭 수행하여야 함! (사실 0번 말고 예측가격이 있는 ROW의 데이터를 써야 함...)
def inv_transform(scaler, data):
    dummy = pd.DataFrame(np.zeros((len(data), scaler.n_features_in_)))
    dummy[0] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=dummy.columns)
    return dummy[0].values


predictions = inv_transform(scaler, predictions)

y_test = list(y_test)

# 예측 결과 출력
for i in range(10):
    print('실제 가격:', y_test[i], '예측 가격:', predictions[i])
