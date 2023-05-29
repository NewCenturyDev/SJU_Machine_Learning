import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 데이터 추출 경로
PREPED_DATA_IN_PATH = './data/preped/'
DATA_OUT_PATH = './result/'

# Pandas DF print시 표시 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)
pd.set_option('display.width', 1000)


def parse_datetime(data):
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d_%H')
    data.set_index('datetime', inplace=True)


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


def convert_dtypes(data):
    # 어차피 이미 분류되어 있음으로 제거
    del data['weekend']
    data['vacation'] = data['vacation'].astype('float64')
    data['season'] = data['season'].astype('float64')
    data['templeture'] = data['templeture'].astype('float64')
    data['windspeed'] = data['windspeed'].astype('float64')
    data['humidity'] = data['humidity'].astype('float64')


def separate_feature_and_label(data):
    # 모든 행, 마지막 열을 제외한 열 추출 (특성  추출)
    features = data.iloc[:, :-1]
    # 모든 행, 마지막 열 추출 (라벨 추출)
    labels = data.iloc[:, -1:]
    return features, labels


# 정규화(스케일링) 진행
# ms = MinMaxScaler()
# ss = StandardScaler()
#
# features_ss = ss.fit_transform(features)
# labels_ms = ms.fit_transform(labels)

# 학습, 검증 데이터 분리
def do_train(features, labels):
    random_seed = 7
    x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.3, random_state=random_seed)

    print("Training Shape", x_train.shape, y_train.shape)
    print("Testing Shape", x_valid.shape, y_valid.shape)

    # Dataframe을 numpy array로 변환
    x_train = np.asarray(x_train.values).astype('float32')
    x_valid = np.asarray(x_valid.values).astype('float32')
    y_train = np.asarray(y_train.values).astype('float32')
    y_valid = np.asarray(y_valid.values).astype('float32')

    print("Training Shape", x_train.shape, y_train.shape)
    print("Testing Shape", x_valid.shape, y_valid.shape)

    # 시계열 데이터로 변환 (window 변경 가능)
    seq_length = 24
    x_train, y_train = split_sequences(x_train, y_train, seq_len=seq_length, steps=1)
    x_valid, y_valid = split_sequences(x_valid, y_valid, seq_len=seq_length, steps=1)
    print(x_train)

    print("========== 조합된 학습 데이터 형식 ==========")
    print(x_train.shape)

    # 하이퍼 파라미터 설정

    print("========== 학습 시작 ==========")
    model = Sequential()
    model.add(Bidirectional(LSTM(10, activation='relu', input_shape=(seq_length, 6)), merge_mode='concat'))
    model.add(Dense(1, activation='linear'))
    # 인자를 설정 (제곱오차 기준 회귀추정 사용, root_mean_square_error - 평균 제곱 오차 사용)
    model.compile(
        loss='mean_squared_error', optimizer='adam', metrics=[
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    # 500회 학습 수행
    model.fit(x_train, y_train, epochs=500, validation_data=(x_valid, y_valid), batch_size=128)


    return model


def do_predict(weekday_model, weekend_model, weekday_f, weekend_f):
    # 정규화 수행
    # df_x_ss = ss.transform(data.iloc[:, :-1])
    # df_y_ms = ms.transform(data.iloc[:, -1:])

    # df_x_ss = Variable(torch.Tensor(features))

    weekday_x_test_tensors = Variable(torch.Tensor(weekday_f.to_numpy()))
    weekend_x_test_tensors = Variable(torch.Tensor(weekend_f.to_numpy()))

    # 시계열 데이터로 재구성
    weekday_x_test = torch.reshape(
        weekday_x_test_tensors, (weekday_x_test_tensors.shape[0], 1, weekday_x_test_tensors.shape[1])
    )
    weekend_x_test = torch.reshape(
        weekend_x_test_tensors, (weekend_x_test_tensors.shape[0], 1, weekend_x_test_tensors.shape[1])
    )

    # 테스트 진행
    weekday_test_predict = weekday_model.eval(weekday_x_test)
    weekend_test_predict = weekend_model.eval(weekend_x_test)

    # 정규화 해제
    # predicted = ms.inverse_transform(predicted)
    # label_y = ms.inverse_transform(label_y)

    # 결과 취합
    weekday_output = pd.DataFrame({'Time': weekday_f['datetime'], 'Power': weekday_test_predict})
    weekend_output = pd.DataFrame({'Time': weekend_f['datetime'], 'Power': weekend_test_predict})
    result = pd.concat([weekday_output, weekend_output], ignore_index=True)
    result.sort_values('Time', inplace=True)

    # CSV 파일로 저장
    result.to_csv(DATA_OUT_PATH + 'bi_lstm_seperated.csv', index=False)


def plot_result(valid_y, predict_y):
    # 결과 그래프 표출
    plt.figure(figsize=(10, 6))
    plt.axvline(x=200, c='r', linestyle='--')
    plt.plot(valid_y, label='Actual Data')
    plt.plot(predict_y, label='Predicted Data')
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()


# MAIN
# 전처리된 데이터 Pandas DataFrame이 들어있는 CSV 읽기
weekday_data = pd.read_csv(PREPED_DATA_IN_PATH + 'train_with_weekday.csv')
weekend_data = pd.read_csv(PREPED_DATA_IN_PATH + 'train_with_weekend.csv')
parse_datetime(weekday_data)
parse_datetime(weekend_data)
convert_dtypes(weekday_data)
convert_dtypes(weekend_data)
print(weekday_data.dtypes)
print(weekend_data.dtypes)

weekday_feature, weekday_label = separate_feature_and_label(weekday_data)
weekend_feature, weekend_label = separate_feature_and_label(weekend_data)

weekday_bi_lstm_model = do_train(weekday_feature, weekday_label)
weekend_bi_lstm_model = do_train(weekend_feature, weekend_label)


test_data = pd.read_csv(PREPED_DATA_IN_PATH + 'test.csv')
# 주말/평일 구분
weekday_test_data = test_data[~test_data['weekend']]
weekend_test_data = test_data[test_data['weekend']]
parse_datetime(weekday_test_data)
parse_datetime(weekend_test_data)
convert_dtypes(weekday_test_data)
convert_dtypes(weekend_test_data)
do_predict(weekday_bi_lstm_model, weekend_bi_lstm_model, weekday_test_data, weekend_test_data)
