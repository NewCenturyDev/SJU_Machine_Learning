from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import numpy as np

# 데이터 불러오기 - 인종차별 이슈로 사이킷런에서 삭제. openml에서 가져옴
# boston = load_boston()
# 예측 변수와 목표 변수를 각각 X, y에 저장
x, y = datasets.fetch_openml('boston', return_X_y=True)
# 데이터셋을 학습 데이터셋과 테스트 데이터셋으로 분할
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Linear Regressor 객체 생성
lin_reg = LinearRegression()
# 모델 학습
lin_reg.fit(x_train, y_train)
# 테스트 데이터셋으로 예측 수행
y_pred = lin_reg.predict(x_test)
# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regressor RMSE:", rmse)
