import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 맷플랏립 GUI 백엔드 설정
matplotlib.use('tkagg')

# 경로 설정
TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/test.csv'
RESULT_OUT_PATH = './data/'

# Pandas 모든 열 보이게 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def load_titanic_data():
    # 타이타닉 데이터 읽기
    # train_df = 학습용 데이터
    # test_df = 테스트용 데이터 (성능 평가용 데이터)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return {
        'train': train_df,
        'test': test_df
    }


def plot_variable_relavents(raw_data):
    # 수치형 데이터 간의 산점도 그리기
    # 컴퓨터 사양에 따라 표출이 오래 걸릴 수 있음
    sns.set(font_scale=0.7)
    sns.set_style('ticks')
    # 항구 코드는 색깔로 처리

    graph = sns.pairplot(raw_data, diag_kind=None, hue='Embarked')
    graph.fig.set_figheight(8)
    graph.fig.set_figwidth(12)
    graph.fig.subplots_adjust(bottom=0.06)
    plt.show()


def plot_variable_relavents_with_survival(raw_data, feature):
    # 생존여부와 각 변수의 상관관계를 알기 위해 그래프로 표현
    # 특정 변수와 생존여부만 추출
    survived_and_feature = raw_data[[feature, 'Survived']]
    # 각 특성에 따른 생존확률(%) 도출
    survived_prob_by_feature = survived_and_feature.groupby(feature).mean() * 100
    # 생존자만 필터링
    only_survived_feature = survived_and_feature.loc[survived_and_feature['Survived'] == 1]
    # 각 특성에 따른 생존자 수 도출
    survived_cnt_by_feature = only_survived_feature.groupby(feature).count()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    survived_cnt_by_feature['Survived'].plot(kind='bar', color='orange', alpha=0.7, ax=ax, width=0.2, position=1)
    survived_prob_by_feature['Survived'].plot(kind='bar', color='skyblue', alpha=0.7, ax=ax2, width=0.2, position=0)
    ax.set_ylabel('Survived Counts')
    ax2.set_ylabel('Survival Rate')
    ax2.set_ylim([0, 100])
    plt.xlabel(feature)
    ax.legend(['Survived Counts'], loc='upper left')
    ax2.legend(['Survival Rate'], loc='upper right')
    plt.title('Relavence of {} to Survival'.format(feature))
    plt.show()


def analyse_data(train_data):
    # 데이터의 형태를 분석한다
    # PassengerId: 승객번호, Survived: 생존여부, Pclass: 티켓 등급, Name: 이름, Sex: 성별, Age: 나이, SibSp: 타이타닉에 같이 탄 형제자매 또는 아내의 수
    # Parch: 타이타닉에 같이 탄 자녀 또는 부모의 수, Ticket: 티켓 번호, Fare: 지불한 운임, Cabin: 방 번호, Embarked: 승선한 항구 코드

    # 데이터 컬럼의 형태 확인
    print('--------------------------------- 데이터 컬럼의 형태 ---------------------------------')
    print(train_data.info())
    print('---------------------------------------------------------------------------------')

    # 정수 또는 실수 타입의 데이터 컬럼의 통계 정보 확인
    print('-------------------------------- 데이터 컬럼의 통계정보 --------------------------------')
    print(train_data.describe())
    print('---------------------------------------------------------------------------------')

    # 성별에 대한 데이터 통계 정보 산출
    gender_statistics = pd.DataFrame({
        'Sex': ['Male', 'Female'],
        'ServivedCnt': [
            train_data.loc[(train_data['Survived'] == 1) & (train_data['Sex'] == 'male')].count()['Sex'],
            train_data.loc[(train_data['Survived'] == 1) & (train_data['Sex'] == 'female')].count()['Sex']
        ],
        'TotalCnt': [
            train_data.loc[train_data['Sex'] == 'male'].count()['Sex'],
            train_data.loc[train_data['Sex'] == 'female'].count()['Sex']
        ],
    })
    gender_statistics['SurvivalRate'] = [
        gender_statistics.loc[gender_statistics['Sex'] == 'Male']['ServivedCnt'] /
        gender_statistics.loc[gender_statistics['Sex'] == 'Male']['TotalCnt'] * 100,
        gender_statistics.loc[gender_statistics['Sex'] == 'Female']['ServivedCnt'] /
        gender_statistics.loc[gender_statistics['Sex'] == 'Female']['TotalCnt'] * 100
    ]
    print('---------------------------- 성별에 대한 데이터 통계 정보 산출 ---------------------------')
    print(gender_statistics)
    print('---------------------------------------------------------------------------------')

    # 승선한 항구 코드에 대한 데이터 통계 정보 산출
    embarked_statistics = pd.DataFrame({
        'Embarked': ['C', 'Q', 'S'],
        'ServivedCnt': [
            train_data.loc[(train_data['Survived'] == 1) & (train_data['Embarked'] == 'C')].count()['Embarked'],
            train_data.loc[(train_data['Survived'] == 1) & (train_data['Embarked'] == 'Q')].count()['Embarked'],
            train_data.loc[(train_data['Survived'] == 1) & (train_data['Embarked'] == 'S')].count()['Embarked']
        ],
        'TotalCnt': [
            train_data.loc[train_data['Embarked'] == 'C'].count()['Embarked'],
            train_data.loc[train_data['Embarked'] == 'Q'].count()['Embarked'],
            train_data.loc[train_data['Embarked'] == 'S'].count()['Embarked']
        ],
    })
    embarked_statistics['SurvivalRate'] = [
        embarked_statistics.loc[embarked_statistics['Embarked'] == 'C']['ServivedCnt'] /
        embarked_statistics.loc[embarked_statistics['Embarked'] == 'C']['TotalCnt'] * 100,
        embarked_statistics.loc[embarked_statistics['Embarked'] == 'Q']['ServivedCnt'] /
        embarked_statistics.loc[embarked_statistics['Embarked'] == 'Q']['TotalCnt'] * 100,
        embarked_statistics.loc[embarked_statistics['Embarked'] == 'S']['ServivedCnt'] /
        embarked_statistics.loc[embarked_statistics['Embarked'] == 'S']['TotalCnt'] * 100
    ]
    print('------------------------- 승선한 항구 코드에 대한 데이터 통계 정보 산출 -------------------------')
    print(embarked_statistics)
    print('--------------------------------------------------------------------------------------')


def visualize_data_analysis(train_data):
    # 각각의 승객 정보 및 생존 여부들에 대해 상호간의 산점도(경향성)를 시각화하여 표시
    plot_variable_relavents(train_data)
    plot_variable_relavents_with_survival(train_data, 'Pclass')
    plot_variable_relavents_with_survival(train_data, 'Sex')
    plot_variable_relavents_with_survival(train_data, 'Age')
    plot_variable_relavents_with_survival(train_data, 'SibSp')
    plot_variable_relavents_with_survival(train_data, 'Parch')
    plot_variable_relavents_with_survival(train_data, 'Ticket')
    plot_variable_relavents_with_survival(train_data, 'Fare')
    plot_variable_relavents_with_survival(train_data, 'Cabin')
    plot_variable_relavents_with_survival(train_data, 'Embarked')

    # 나이의 경우 조금 더 명확하게 하기 위해 구간별 분포로 다시 히스토그램을 그려보았다
    # 16개 급간으로 분류하였다 (0~80세임으로 대략 5살 단위로 나눠질 것임)
    graph = sns.FacetGrid(train_data, col='Survived', margin_titles=True)
    graph.map(plt.hist, 'Age', bins=16, color='orange')
    graph.set_ylabels('Counts')
    graph.set_xlabels('Age')
    plt.show()


def make_age_band(data_df):
    age_bands = []
    # 나이 변수를 연령대 변수로 묶어주어 범주형 데이터로 변환시킨다
    # 연령대는 5살 단위로 끊었으며, 75살 이상은 전부 15급간으로 취급하였다
    for age in data_df['Age']:
        if age >= 75:
            age_bands.append(15)
        else:
            age_bands.append(int(age / 5))
    data_df['Ageband'] = age_bands
    data_df = data_df.drop('Age', axis=1)
    return data_df


def preprocess_data(dataset):
    # 데이터를 전처리한다
    # 불필요한 데이터 제거 (Ticket, Fare, Cabin, Name, PassengerId)
    dataset['train'] = dataset['train'].drop(['Ticket', 'Fare', 'Cabin', 'Name', 'PassengerId', 'Survived'], axis=1)
    dataset['test'] = dataset['test'].drop(['Ticket', 'Fare', 'Cabin', 'Name', 'PassengerId'], axis=1)

    # sci-kit 라이브러리의 RFC 모델은 정수와 실수만 인식할 수 있다. 따라서 string 또는 object로 된 범주형 데이터는 정수로 인코딩을 해주어야 한다
    # 성별 변수의 자료형을 바꿔주어 sci-kit RFC 모델이 인식할 수 있게 함 (0은 남성, 1은 여성으로 인코딩)
    dataset['train']['Sex'] = dataset['train']['Sex'].map({'male': 0, 'female': 1})
    dataset['test']['Sex'] = dataset['test']['Sex'].map({'male': 0, 'female': 1})

    # 결측치 처리 - 승선항구(Embarked) - S가 제일 많음으로 S로 채워주었다
    dataset['train']['Embarked'] = dataset['train']['Embarked'].fillna('S')
    dataset['test']['Embarked'] = dataset['test']['Embarked'].fillna('S')
    # 승선항구 코드 변수의 자료형을 바꿔주어 sci-kit RFC 모델이 인식할 수 있게 함 (0은 C항, 1은 Q항, 2는 S항으로 인코딩)
    dataset['train']['Embarked'] = dataset['train']['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    dataset['test']['Embarked'] = dataset['test']['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # 결측치 처리 - 나이(Age) - 승객들의 평균 나이로 채워주었다 (단, 정수값으로)
    dataset['train']['Age'] = dataset['train']['Age'].fillna(int(dataset['train']['Age'].mean()))
    dataset['test']['Age'] = dataset['test']['Age'].fillna(int(dataset['test']['Age'].mean()))

    # 나이 변수의 경우 지나치게 파편화 되어 있다 (한살한살을 서로 다른 범주로 놓고 비교하면 곤란하다)
    # 그럼으로, 5살 단위로 묶어주도록 하였다
    dataset['train'] = make_age_band(dataset['train'])
    dataset['test'] = make_age_band(dataset['test'])

    # 동반한 부모자식/아내/남편/형제자매의 수를 합쳐 "동반자 수" 라는 하나의 칼럼으로 정리
    # 이는 생존여부와 SibSp, Parch의 연관관계를 나타낸 히스토그램을 보았을 때 유사한 경향을 띄었기 때문에
    # 연산의 복잡도를 줄이기 위해서 하나의 특성값으로 정리하였다
    dataset['train']['Companion'] = dataset['train']['SibSp'] + dataset['train']['Parch']
    dataset['test']['Companion'] = dataset['test']['SibSp'] + dataset['test']['Parch']
    dataset['train'] = dataset['train'].drop(['SibSp', 'Parch'], axis=1)
    dataset['test'] = dataset['test'].drop(['SibSp', 'Parch'], axis=1)

    # 전처리된 데이터 출력 (훈련용 데이터)
    print('-------------------------- 전처리된 데이터 출력 (훈련용 데이터) --------------------------')
    print(dataset['train'])
    print('---------------------------------------------------------------------------------')
    return {
        'train': dataset['train'],
        'test': dataset['test']
    }


def train_random_forest(train_data, survived_label):
    # 랜덤 포레스트 모델을 이용하여 데이터를 학습한다
    # 시드값 설정 - 잘되라는 의미에서 럭키세븐 두 개에 해당하는 77로 선택
    random_seed = 77

    # 학습 데이터와 검증 데이터 분리 (검증 데이터:학습 데이터 = 25%:75% - 기본값 사용, 랜덤시드는 rfc와 동일하게 고정)
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, survived_label, random_state=random_seed)

    # 모델 생성 및 학습 진행
    print('학습 시작')
    rfc = RandomForestClassifier(random_state=random_seed)
    rfc.fit(x_train, y_train)

    # 검증 데이터를 통해 검증
    y_predict = rfc.predict(x_valid)

    # 학습 결과 출력
    print('------------------------ 검증 데이터를 이용한 데이터 정확도 측정 ------------------------')
    print('Accuracy: {}'.format(accuracy_score(y_valid, y_predict)))
    print('Model Training Report')
    print(classification_report(y_predict, y_valid))
    print('-------------------------------------------------------------------------------')
    return rfc


def validate_with_test_data(rfc, preped_test_data):
    # 라벨이 없는 주어진 데이터와 학습된 랜덤포레스트 모델을 이용해 최종 예측 결과 도출
    return rfc.predict(preped_test_data)


def save_result_to_csv(passenger_id, test_predicted):
    # 최종 예측 결과를 도출
    answer_dataset = pd.DataFrame({'PassengerId': passenger_id, 'Survived': test_predicted})
    answer_dataset.to_csv(RESULT_OUT_PATH + 'titanic_pred_result.csv', index=False, quoting=3)


# 메인 로직
data = load_titanic_data()
analyse_data(data['train'])
visualize_data_analysis(data['train'])
label = data['train']['Survived']
test_id = data['test']['PassengerId']
preped_data = preprocess_data(data)
rfc_model = train_random_forest(preped_data['train'], label)
prediction_result = validate_with_test_data(rfc_model, data['test'])
save_result_to_csv(test_id, prediction_result)
