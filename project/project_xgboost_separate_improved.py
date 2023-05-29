import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from collections import deque

import xgboost as xgb  # XGBoost 모델
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


def parse_date(d):
    return dt.datetime.strptime(d.split('_')[0], "%Y%m%d")


def move_list_for_sequencial(energy_usages_list, step):
    eu_deque = deque(energy_usages_list)
    eu_deque.rotate(step)
    return list(eu_deque)


def load_train_data():
    energy_usage_dataset = pd.read_csv(PREPED_DATA_IN_PATH + 'train.csv')
    return pd.DataFrame({
        'year': list(map(
            lambda d: int(parse_date(d).year), energy_usage_dataset['datetime']
        )),
        'month': list(map(
            lambda d: int(parse_date(d).month), energy_usage_dataset['datetime']
        )),
        'date': list(map(
            lambda d: int(parse_date(d).day), energy_usage_dataset['datetime']
        )),
        'time': list(map(
            lambda t: int(t.split('_')[1]), energy_usage_dataset['datetime']
        )),
        'weekend': list(energy_usage_dataset['weekend']),
        'vacation': list(energy_usage_dataset['vacation']),
        'energy_prev_1': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -1),
        'energy_prev_2': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -2),
        'energy_prev_3': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -3),
        'energy_prev_4': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -4),
        'energy_prev_5': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -5),
        'energy_prev_6': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -6),
        # 'energy_prev_7': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -7),
        # 'energy_prev_8': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -8),
        # 'energy_prev_9': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -9),
        # 'energy_prev_10': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -10),
        # 'energy_prev_11': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -11),
        # 'energy_prev_12': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -12),
        # 'energy_prev_13': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -13),
        # 'energy_prev_14': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -14),
        # 'energy_prev_15': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -15),
        # 'energy_prev_16': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -16),
        # 'energy_prev_17': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -17),
        # 'energy_prev_18': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -18),
        # 'energy_prev_19': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -19),
        # 'energy_prev_20': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -20),
        # 'energy_prev_21': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -21),
        # 'energy_prev_22': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -22),
        # 'energy_prev_23': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), -23),
        'energy_next_1': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 1),
        'energy_next_2': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 2),
        'energy_next_3': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 3),
        'energy_next_4': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 4),
        'energy_next_5': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 5),
        'energy_next_6': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 6),
        # 'energy_next_7': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 7),
        # 'energy_next_8': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 8),
        # 'energy_next_9': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 9),
        # 'energy_next_10': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 10),
        # 'energy_next_11': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 11),
        # 'energy_next_12': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 12),
        # 'energy_next_13': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 13),
        # 'energy_next_14': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 14),
        # 'energy_next_15': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 15),
        # 'energy_next_16': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 16),
        # 'energy_next_17': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 17),
        # 'energy_next_18': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 18),
        # 'energy_next_19': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 19),
        # 'energy_next_20': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 20),
        # 'energy_next_21': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 21),
        # 'energy_next_22': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 22),
        # 'energy_next_23': move_list_for_sequencial(list(energy_usage_dataset['energy_usage']), 23),
        'energy_usage': list(energy_usage_dataset['energy_usage']),
    })


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


def load_test_data(interpolated_train_data):
    energy_usage_testset = pd.read_csv(DATA_IN_PATH + 'Answer_sheet.csv')
    time_parsed_test_data = pd.DataFrame({
        'year': list(map(
            lambda d: int(parse_date(d).year), energy_usage_testset['Time']
        )),
        'month': list(map(
            lambda d: int(parse_date(d).month), energy_usage_testset['Time']
        )),
        'date': list(map(
            lambda d: int(parse_date(d).day), energy_usage_testset['Time']
        )),
        'time': list(map(
            lambda t: int(t.split('_')[1]), energy_usage_testset['Time']
        )),
    })

    # Attach Spline Interpolated/Nearby Sequencial Data
    # 주변 시간대 전력값(앞뒤 몇시간) 불러오기
    # (단, 결측치가 연속한 경우 해당 부분은 3차스플라인 interpolation 된 데이터 사용
    merged_df = pd.merge(
        time_parsed_test_data, interpolated_train_data, left_on=['year', 'month', 'date', 'time'],
        right_on=['year', 'month', 'date', 'time'], how='left'
    )
    # 주변 시간대 전력량을 가져오기 위해서 딸려온 라벨값 + 단순 3차 스플라인 보정한 초기값 삭제
    del merged_df['energy_usage']
    print(merged_df)
    return merged_df


def visualize_parameters(model, col_names):
    # 중요 변수 시각화
    plot_importance(model)
    plt.yticks(range(len(col_names)), col_names)
    plt.show()


def do_train(data_df, energy_usages):
    features = data_df
    print("========== 조합된 학습 데이터 예시 ==========")
    print(features)

    # 학습 데이터에서 검증 데이터 분리
    x_train, x_valid, y_train, y_valid = train_test_split(features, energy_usages, test_size=0.3, random_state=7)

    print("========== 조합된 학습 데이터 형식 ==========")
    print(x_train.shape)
    print(x_train.dtypes)
    print("========== 라벨 개수 ==========")
    print(len(y_train))

    # 모델 생성 및 파라미터 설정 (회귀트리 예측 사용, root_mean_square_error - 평균 제곱 오차 사용)
    # 파라미터 설정 근거 - http://journal.auric.kr/kiee/XmlViewer/f387530
    # 위 논문 참고하였음
    train_data_dmatrix = xgb.DMatrix(x_train, y_train)
    eval_data_dmatrix = xgb.DMatrix(x_valid, y_valid)
    data_list = [(train_data_dmatrix, 'train'), (eval_data_dmatrix, 'valid')]

    # 인자를 설정 (제곱오차 기준 회귀추정 사용, root_mean_square_error - 평균 제곱 오차 사용)
    # 주석친 것은 적용하지 않는 게 성능면에서 더 나음
    params = {
        'eta': 0.005,
        'max_depth': 8,
        'max_leaves': 250,
        'subsample': 0.90,
        'colsample_bytree': 1,
        'tree_method': 'exact',
        'min_child_weight': 1,
        'gamma': 0.01,
        'seed': 7,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
    }

    # 모델 학습
    print("========== 학습 시작 ==========")
    model = xgb.train(params, train_data_dmatrix, num_boost_round=50000, evals=data_list, early_stopping_rounds=50)

    # 중요 파라미터 시각화
    # visualize_parameters(model, [
    #     'year', 'month', 'date', 'time', 'vacation', 'weekend', 'templeture', 'windspeed', 'humidity', 'rainfall'
    # ])

    return model


def do_test(weekday_model, weekend_model, data):
    weekday_data = data[~data['weekend']]
    weekend_data = data[data['weekend']]

    weekday_features = xgb.DMatrix(weekday_data)
    weekend_features = xgb.DMatrix(weekend_data)

    # 학습된 모델로 테스트 진행
    weekday_test_predict = weekday_model.predict(weekday_features)
    weekend_test_predict = weekend_model.predict(weekend_features)

    # 예측 결과물 출력 디렉터리 생성
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    # 시계열 데이터를 다시 연월일_시 타임스탬프 스트링으로 조합
    datetime_columns = ['year', 'month', 'date', 'time']

    weekday_timestamps = weekday_data[datetime_columns].apply(
        lambda record: '{0}{1:02d}{2:02d}_{3}'.format(record[0], record[1], record[2], record[3]), axis=1
    )
    weekend_timestamps = weekend_data[datetime_columns].apply(
        lambda record: '{0}{1:02d}{2:02d}_{3}'.format(record[0], record[1], record[2], record[3]), axis=1
    )

    # 결과 취합
    weekday_output = pd.DataFrame({'Time': weekday_timestamps, 'Power': weekday_test_predict})
    weekend_output = pd.DataFrame({'Time': weekend_timestamps, 'Power': weekend_test_predict})
    result = pd.concat([weekday_output, weekend_output], ignore_index=True)
    result.sort_values('Time', inplace=True)

    # CSV 파일로 저장
    result.to_csv(DATA_OUT_PATH + 'simple_xgb_seperate_improved.csv', index=False)


# MAIN
train_dataset = load_train_data()

# 이상치(아웃라이어) 제거 => 0인 데이터들과, 1000 넘는 데이터 (1건) 제거
cleaned_train_dataset = train_dataset.drop(
    train_dataset[train_dataset.energy_usage == 0].index
)
cleaned_train_dataset = cleaned_train_dataset.drop(
    cleaned_train_dataset[cleaned_train_dataset.energy_usage > 1000].index
)


train_with_weather_df = load_weather_data(cleaned_train_dataset)
train_with_weekday_df = train_with_weather_df[~train_with_weather_df['weekend']]
train_with_weekend_df = train_with_weather_df[train_with_weather_df['weekend']]

# 라벨 추출
weekday_labels = train_with_weekday_df['energy_usage']
weekend_labels = train_with_weekend_df['energy_usage']
del train_with_weekday_df['energy_usage']
del train_with_weekend_df['energy_usage']

weekday_xg_boost_model = do_train(train_with_weekday_df, weekday_labels)
weekend_xg_boost_model = do_train(train_with_weekend_df, weekend_labels)
test_df = load_test_data(train_dataset)
test_with_weather_df = load_weather_data(test_df)
do_test(weekday_xg_boost_model, weekend_xg_boost_model, test_with_weather_df)
