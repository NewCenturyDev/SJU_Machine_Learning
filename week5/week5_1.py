import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 맷플랏립 GUI 백엔드 설정
matplotlib.use('tkagg')


def load_and_analyze():
    # Moons 데이터셋 로드
    x, y = make_moons(n_samples=500, noise=0.3, random_state=42)

    # 산점도 그리기
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, palette='rainbow')
    plt.title('Scatter plot of original Moons dataset')
    plt.show()

    # 원본 데이터 첫 번째 특성에 대한 히스토그램
    plt.hist(x[:, 0], bins=25)
    plt.title('Histogram of feature 1 of original Moons dataset')
    plt.show()

    # 첫 번쨰 특성을 제곱한 값에 대한 히스토그램
    plt.hist(x[:, 0] ** 2, bins=25)
    plt.title('Histogram of feature 1 squared of Moons dataset')
    plt.show()
    return {
        'x': x,
        'y': y
    }


def run_svm_model(x, y):
    # 학습 데이터와 테스트 데이터 분리
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    # 선형 SVM 모델 생성 및 학습
    svm_linear = SVC(kernel='linear', C=1, gamma='auto')
    svm_linear.fit(x_train, y_train)

    # 비선형 SVM 모델 생성 및 학습
    svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1)
    svm_rbf.fit(x_train, y_train)

    # 테스트 데이터를 학습된 모델에 넣어 예측 및 정확도 평가
    y_pred_linear = svm_linear.predict(x_test)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)

    y_pred_rbf = svm_rbf.predict(x_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

    print('Linear SVM accuracy: %.2f' % accuracy_linear)
    print('RBF SVM accuracy: %.2f' % accuracy_rbf)


# 메인 로직
data = load_and_analyze()
run_svm_model(data['x'], data['y'])
