from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 숫자 데이터셋 로드
digits = load_digits()
# digits.data = 입력 데이터, digits.target = 라벨
X, y = digits.data, digits.target

# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습 및 예측
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

# 성능 평가
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
