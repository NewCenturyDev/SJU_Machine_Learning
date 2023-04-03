import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 맷플랏립 GUI 백엔드 설정
matplotlib.use('tkagg')

# 경로 설정
TRAIN_DATA_PATH = './data/train.csv'


def load_titanic_data():
    # 타이타닉 데이터 읽기
    # passengerId: 승객번호, Survived: 생존여부, Pclass: 티켓 등급, sex: 성별, age: 나이, sibsp: 타이타닉에 같이 탄 형제자매 또는 아내의 수
    # parch: 타이타닉에 같이 탄 자녀 또는 부모의 수, ticket: 티켓 번호, fare: 지불한 운임, cabin: 방 번호, embarked: 승선한 항구 코드
    data_file = pd.read_csv(TRAIN_DATA_PATH)
    return {
        'PassengerId': data_file['PassengerId'],
        'Survived': data_file['PassengerId'],
        'Pclass': data_file['PassengerId'],
        'Name': data_file['PassengerId'],
        'Sex': data_file['PassengerId'],
        'Age': data_file['PassengerId'],
        'SibSp': data_file['PassengerId'],
        'Parch': data_file['PassengerId'],
        'Ticket': data_file['PassengerId'],
        'Fare': data_file['PassengerId'],
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
data = load_titanic_data()
run_svm_model(data['x'], data['y'])
