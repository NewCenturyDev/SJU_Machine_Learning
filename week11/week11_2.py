import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 데이터 불러오기 - 인종차별 이슈로 사이킷런에서 삭제. openml에서 가져옴
# boston = load_boston()
x, y = datasets.fetch_openml('boston', return_X_y=True)

# 데이터 스케일링 (매우 중요 - 이것을 수행하지 않아서 이상한 값이 자주 도출)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# 학습 데이터와 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# 모델 학습
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(x_train, y_train)

# 테스트 데이터 예측
y_pred = model.predict(x_test)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
