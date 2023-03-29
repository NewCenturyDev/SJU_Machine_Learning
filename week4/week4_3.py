import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 붓꽃 데이터셋 로드
iris = load_iris()
X, y = iris.data[:, 1:3], iris.target

# train-test split ratio (테스트 데이터 비율을 0.1~0.9까지 각각 선택 - 9개로 쪼갬 즉 0.1단위)
ratios = np.linspace(0.1, 0.9, num=9)

# 모델별 정확도를 저장할 배열 선언
nb_accs, rf_accs = [], []

for ratio in ratios:
    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=41)

    # 나이브 베이즈 분류 모델 생성 및 학습
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # 랜덤 포레스트 분류 모델 생성 및 학습
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 각 모델의 예측 결과
    nb_pred = nb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # 각 모델의 정확도 계산 및 저장
    nb_accuracy = accuracy_score(y_test, nb_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    nb_accs.append(nb_accuracy)
    rf_accs.append(rf_accuracy)

# 결과 그래프 출력
pyplot =
plt.plot(ratios, nb_accs, label="Naive Bayes")
plt.plot(ratios, rf_accs, label="Random Forest")
plt.title('Accuracy by test set ratio')
plt.xlabel('Test set ratio')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
