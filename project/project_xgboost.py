import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split  # 테스트/훈련 데이터 분리
from xgboost import XGBRegressor  # 회귀트리 모델
from xgboost import plot_importance  # 중요변수 시각화

matplotlib.use('TkAgg')

DATA_IN_PATH = './data/'
PREPED_DATA_IN_PATH = './data/preped/'
DATA_OUT_PATH = './result/'


def load_train_data():
    energy_usage_dataset = pd.read_csv(DATA_IN_PATH + 'Train_dataset.csv')
    return {
        'year': list(map(
            lambda d: int(dt.datetime.strptime(d.split('_')[0], "%Y%m%d").year), energy_usage_dataset['Time']
        )),
        'month': list(map(
            lambda d: int(dt.datetime.strptime(d.split('_')[0], "%Y%m%d").month), energy_usage_dataset['Time']
        )),
        'date': list(map(
            lambda d: int(dt.datetime.strptime(d.split('_')[0], "%Y%m%d").day), energy_usage_dataset['Time']
        )),
        'time': list(map(
            lambda t: int(t.split('_')[1]), energy_usage_dataset['Time']
        )),
        'energy_usage': list(energy_usage_dataset['Power']),
    }


def load_test_data():
    energy_usage_testset = pd.read_csv(DATA_IN_PATH + 'Answer_sheet.csv')
    return {
        'year': list(map(
            lambda d: int(dt.datetime.strptime(d.split('_')[0], "%Y%m%d").year), energy_usage_testset['Time']
        )),
        'month': list(map(
            lambda d: int(dt.datetime.strptime(d.split('_')[0], "%Y%m%d").month), energy_usage_testset['Time']
        )),
        'date': list(map(
            lambda d: int(dt.datetime.strptime(d.split('_')[0], "%Y%m%d").day), energy_usage_testset['Time']
        )),
        'time': list(map(
            lambda t: int(t.split('_')[1]), energy_usage_testset['Time']
        )),
    }


def visualize_parameters(model, col_names):
    # 중요 변수 시각화
    plot_importance(model)
    plt.yticks(range(4), col_names)
    plt.show()


def do_train(year, month, date, time, energy_usage):
    features = pd.DataFrame({
        'year': year,
        'month': month,
        'date': date,
        'time': time,
    })
    print(features)
    print(energy_usage)

    # 학습 데이터에서 검증 데이터 분리
    x_train, x_valid, y_train, y_valid = train_test_split(features, energy_usage, test_size=0.3, random_state=7)
    print(x_train.shape)
    print(x_train.dtypes)
    print(len(y_train))

    # 모델 생성 및 자를 설정 (회귀트리 예측 사용, root_mean_square_error - 평균 제곱 오차 사용)
    model = XGBRegressor(
        objective='reg:linear',
        eval_metric='rmse',
    )
    # 모델 학습
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)])

    # 중요 파라미터 시각화
    visualize_parameters(model, ['year', 'month', 'date', 'time'])

    return model


def do_test(model, year, month, date, time):
    features = pd.DataFrame({
        'year': year,
        'month': month,
        'date': date,
        'time': time,
    })

    # 학습된 모델로 테스트 진행
    test_predict = model.predict(features)

    # 예측 결과물 출력 디렉터리 생성
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    # 시계열 데이터를 다시 연월일_시 타임스탬프 스트링으로 조합
    datetime_columns = ['year', 'month', 'date', 'time']
    print(features[datetime_columns])

    timestamps = features[datetime_columns].apply(
        lambda record: '{0}{1:02d}{2:02d}_{3}'.format(record[0], record[1], record[2], record[3]), axis=1
    )
    print(timestamps)

    # CSV 파일로 저장
    output = pd.DataFrame({'Time': timestamps, 'Power': test_predict})
    output.to_csv(DATA_OUT_PATH + 'simple_xgb.csv', index=False)


# MAIN
train_dataset = load_train_data()
xg_boost_model = do_train(
    train_dataset['year'], train_dataset['month'], train_dataset['date'], train_dataset['time'],
    train_dataset['energy_usage']
)
test_dataset = load_test_data()
do_test(
    xg_boost_model, test_dataset['year'], test_dataset['month'], test_dataset['date'], test_dataset['time'],
)
