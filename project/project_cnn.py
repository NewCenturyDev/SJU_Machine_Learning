import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np

import tensorflow as tf
import xgboost as xgb  # XGBoost 모델
from sklearn.model_selection import train_test_split  # 테스트/훈련 데이터 분리
from xgboost import plot_importance  # 중요변수 시각화

matplotlib.use('TkAgg')

DATA_IN_PATH = './data/'
PREPED_DATA_IN_PATH = './data/preped/'
DATA_OUT_PATH = './result/'


def load_train_data():
    energy_usage_dataset = pd.read_csv(DATA_IN_PATH + 'Train_dataset.csv')
    energy_usage_df = pd.DataFrame({
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
    })
    return {
        'data': energy_usage_df,
        'energy_usage': list(energy_usage_dataset['Power']),
    }


def load_weather_data(data):
    weather_dataset = pd.read_csv(DATA_IN_PATH + 'weather.csv')
    weather_data_processed_df = pd.DataFrame({
        'year': list(map(
            lambda d: int(dt.datetime.strptime(d.split(' ')[0], "%Y-%m-%d").year), weather_dataset['tm']
        )),
        'month': list(map(
            lambda d: int(dt.datetime.strptime(d.split(' ')[0], "%Y-%m-%d").month), weather_dataset['tm']
        )),
        'date': list(map(
            lambda d: int(dt.datetime.strptime(d.split(' ')[0], "%Y-%m-%d").day), weather_dataset['tm']
        )),
        'time': list(map(
            lambda t: int(t.split(' ')[1].split(':')[0]), weather_dataset['tm']
        )),
        'templeture': list(map(
            lambda ta: int(ta), weather_dataset['ta']
        )),
        'windspeed': list(map(
            lambda ws: int(ws), weather_dataset['ws']
        )),
        'humidity': list(map(
            lambda hm: int(hm), weather_dataset['hm']
        )),
        'rainfall': list(map(
            lambda rf: float(0.0 if math.isnan(rf) else rf), weather_dataset['rn']
        )),
    })

    # Attach Weather Data
    return pd.merge(
        data, weather_data_processed_df, left_on=['year', 'month', 'date', 'time'],
        right_on=['year', 'month', 'date', 'time'], how='left'
    )


def load_test_data():
    energy_usage_testset = pd.read_csv(DATA_IN_PATH + 'Answer_sheet.csv')
    return pd.DataFrame({
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
    })


def visualize_parameters(model, col_names):
    # 중요 변수 시각화
    plot_importance(model)
    plt.yticks(range(len(col_names)), col_names)
    plt.show()


def do_train(data_df, energy_usage):
    features = data_df
    print("========== 조합된 학습 데이터 예시 ==========")
    print(features)

    # 학습 데이터에서 검증 데이터 분리
    x_train, x_valid, y_train, y_valid = train_test_split(features, energy_usage, test_size=0.3, random_state=7)

    print("========== 조합된 학습 데이터 형식 ==========")
    print(x_train.shape)
    print(x_train.dtypes)
    print("========== 라벨 개수 ==========")
    print(len(y_train))

    # 모델 생성 및 파라미터 설정 (회귀트리 예측 사용, root_mean_square_error - 평균 제곱 오차 사용)
    # 파라미터 설정 근거 - http://journal.auric.kr/kiee/XmlViewer/f387530
    # 위 논문 참고하였음

    # 인자를 설정 (제곱오차 기준 회귀추정 사용, root_mean_square_error - 평균 제곱 오차 사용)
    # 주석친 것은 적용하지 않는 게 성능면에서 더 나음

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='rmse')

    # 모델 학습
    print("========== 학습 시작 ==========")
    model = model.fit(x_train, y_train, epochs=100)

    # 중요 파라미터 시각화
    visualize_parameters(model, ['year', 'month', 'date', 'time', 'templeture', 'windspeed', 'humidity', 'rainfall'])

    return model


def do_test(model, data):
    features = xgb.DMatrix(data)

    # 학습된 모델로 테스트 진행
    test_predict = model.predict(features)

    # 예측 결과물 출력 디렉터리 생성
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    # 시계열 데이터를 다시 연월일_시 타임스탬프 스트링으로 조합
    datetime_columns = ['year', 'month', 'date', 'time']

    timestamps = data[datetime_columns].apply(
        lambda record: '{0}{1:02d}{2:02d}_{3}'.format(record[0], record[1], record[2], record[3]), axis=1
    )

    # CSV 파일로 저장
    output = pd.DataFrame({'Time': timestamps, 'Power': test_predict})
    output.to_csv(DATA_OUT_PATH + 'simple_xgb_improved.csv', index=False)


# MAIN
train_dataset = load_train_data()
train_with_weather_df = load_weather_data(train_dataset['data'])
cnn_model = do_train(train_with_weather_df, train_dataset['energy_usage'])
test_df = load_test_data()
test_with_weather_df = load_weather_data(test_df)
do_test(cnn_model, test_with_weather_df)
