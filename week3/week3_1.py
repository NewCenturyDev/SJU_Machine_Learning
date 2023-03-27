from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터셋 불러오기
iris = load_iris()
print(iris)
# 학습용 데이터와 테스트용 데이터 5: 5로 분할
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=42)
# 의사결정 나무 모델 생성 및 학습
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
# 테스트 데이터 예측 및 정확도 평가
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)
