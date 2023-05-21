import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split  # 테스트/훈련 데이터 분리
from xgboost import plot_importance  # 중요변수 시각화

matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)
pd.set_option('display.width', 1000)

DATA_IN_PATH = './data/'
PREPED_DATA_IN_PATH = './data/preped/'
DATA_OUT_PATH = './result/'


def load_train_data():
    return pd.read_csv(PREPED_DATA_IN_PATH + 'train_combined.csv')


def load_test_data():
    return pd.read_csv(PREPED_DATA_IN_PATH + 'test_combined.csv')


def visualize_parameters(model, col_names):
    # 중요 변수 시각화
    plot_importance(model)
    plt.yticks(range(len(col_names)), col_names)
    plt.show()


def split_sequences(dataset: np.array, powers: np.array, seq_len: int, steps: int) -> tuple:
    # feature와 y 각각 sequential dataset을 반환할 리스트 생성
    x, y = list(), list()
    # sequence length와 step에 따라 sequential dataset 생성
    for i, _ in enumerate(dataset):
        idx_in = i + seq_len
        idx_out = idx_in + steps
        if idx_out > len(dataset):
            break
        seq_x = dataset[i:idx_in]
        seq_y = powers[idx_out - 1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def do_train(data_df, energy_usages):
    # 데이터 확인
    print("========== 데이터 미리보기 ==========")
    print(data_df)

    # 학습 데이터에서 검증 데이터 분리
    x_train, x_valid, y_train, y_valid = train_test_split(data_df, energy_usages, test_size=0.3, random_state=7)

    # Dataframe을 numpy array로 변환
    x_train = np.asarray(x_train.values).astype('float32')
    x_valid = np.asarray(x_valid.values).astype('float32')
    y_train = np.asarray(y_train.values).astype('float32')
    y_valid = np.asarray(y_valid.values).astype('float32')

    # 시계열 데이터로 변환 (window 만들기)
    x_train, y_train = split_sequences(x_train, y_train, 6, 1)
    x_valid, y_valid = split_sequences(x_valid, y_valid, 6, 1)

    # 3차원으로 재구성: 차원 튜플 -> 전체 데이터 수, window 사이즈, feature 수)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 10))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 10))

    print("========== 조합된 학습 데이터 형식 ==========")
    print(x_train.shape)
    print("========== 라벨 개수 ==========")
    print(len(y_train))

    # 모델 생성 및 파라미터 설정 (LSTM 시퀀셜 딥러닝 인공신경망 모델 사용, root_mean_square_error - 평균 제곱 오차 사용)
    # 위 논문 참고하였음

    # 모델 학습
    #     LSTM 네트워크를 생성한 결과를 반환한다.
    #
    #     :param seq_len: Length of sequences. (Look back window size)
    #     :param n_features: Number of features. It requires for model input shape.
    #     :param lstm_units: Number of cells each LSTM layers.
    #     :param learning_rate: Learning rate.
    #     :param dropout: Dropout rate.
    #     :param steps: Length to predict.
    #     :param metrics: Model loss function metric.
    #     :param single_output: Whether 'yhat' is a multiple value or a single value.
    #     :param last_lstm_return_sequences: Last LSTM's `return_sequences`. Allow when `single_output=False` only.
    #     :param dense_units: Number of cells each Dense layers. It adds after LSTM layers.
    #     :param activation: Activation function of Layers.
    # https://www.kais99.org/jkais/journal/Vol21No10/vol21no10p02.pdf
    print("========== 학습 시작 ==========")
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(6, 10)))
    model.add(Dense(1, activation='linear'))
    # 인자를 설정 (제곱오차 기준 회귀추정 사용, root_mean_square_error - 평균 제곱 오차 사용)
    model.compile(
        loss='mean_squared_error', optimizer='adam', metrics=[
            tf.keras.metrics.Accuracy(), tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    # 500회 학습 수행
    model.fit(x_train, y_train, epochs=500, batch_size=1)

    # 중요 파라미터 시각화 (파라미터의 중요도를 알 수 있음)
    visualize_parameters(model, [
        'year', 'month', 'date', 'time', 'vacation', 'weekend', 'templeture', 'windspeed', 'humidity', 'rainfall'
    ])

    return model


def do_test(model, data):
    # 학습된 모델로 테스트 진행
    test_predict = model.predict(data)

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

# 라벨 추출
labels = train_dataset['energy_usage']
del train_dataset['energy_usage']

lstm_model = do_train(train_dataset, labels)
test_dataset = load_test_data()
do_test(lstm_model, test_dataset)
